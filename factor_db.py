"""
因子数据库 - 增强版
管理因子候选库，支持增删改查、批量操作、导出

增强内容：
1. 多状态流转（pending → approved → rejected → active）
2. 批量操作（批量审核、批量删除）
3. 因子去重（同名检查 + 定义相似度检查）
4. 因子合并（多论文的同一因子聚合）
5. 多格式导出（JSON / Markdown / CSV）
6. 因子分组和标签
7. 因子血缘追踪（哪个论文贡献的）
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class FactorRecord:
    """因子记录数据类"""
    factor_name: str = ""
    definition: str = ""
    data_needed: str = ""
    calculation_formula: str = ""
    market_scope: str = "通用"
    paper_conclusion: str = ""
    confidence: str = "中"
    source: str = ""
    paper_link: str = ""
    paper_title: str = ""
    paper_abstract: str = ""

    # 扩展字段
    status: str = "pending"       # pending / approved / rejected / active
    modality: str = "single"      # single / multimodal
    fusion_method: str = ""       # combination / summation / attention / mixture
    news_contribution: str = ""    # 高 / 中 / 低
    news_required: bool = False
    llm_dependency: bool = False
    tags: List[str] = field(default_factory=list)
    quality_score: int = 0       # 0-100

    # 元数据
    added_at: str = ""
    updated_at: str = ""
    approved_at: str = ""
    backtest_result: Dict = field(default_factory=dict)
    notes: str = ""


class FactorDatabase:
    """
    因子数据库管理器（增强版）

    状态流转：
    pending → approved → active（通过审核进入实盘池）
    pending → rejected（审核未通过）
    active → deprecated（实盘发现无效后标记）

    使用示例：
    db = FactorDatabase()
    db.add_factor(factor_dict)
    db.approve_factor("估值因子")
    results = db.get_active_factors()
    db.export_to_markdown("因子报告.md")
    """

    def __init__(self, db_path: Optional[str] = None):
        from config import FACTOR_DB_PATH, DATA_DIR
        self.db_path = db_path or str(FACTOR_DB_PATH)
        self.factors: List[Dict] = []
        self.load()

    # ------------------------------------------------------
    # 基础CRUD
    # ------------------------------------------------------

    def load(self):
        """从文件加载因子库"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    self.factors = data
                elif isinstance(data, dict):
                    self.factors = data.get("factors", [])
                    meta = data.get("meta", {})
                    print(f"已加载 {len(self.factors)} 个因子（更新于{meta.get('updated_at','未知')}）")
                    return

                print(f"已加载 {len(self.factors)} 个因子")
            except Exception as e:
                print(f"加载失败: {e}，将创建新库")
                self.factors = []
        else:
            print("因子库为空，将创建新库")

    def save(self):
        """保存因子库到文件"""
        data = {
            "meta": {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_count": len(self.factors),
            },
            "factors": self.factors,
        }

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已保存 {len(self.factors)} 个因子到 {self.db_path}")

    def add_factor(self, factor: Dict, auto_yes: bool = False) -> bool:
        """
        添加新因子（自动去重 + 状态初始化）

        Args:
            factor: 因子字典
            auto_yes: True 时跳过相似因子的交互确认（agent/批量场景用）

        Returns:
            True=新增成功，False=已存在跳过
        """
        factor_name = factor.get("factor_name", "")

        # 去重检查（同名）
        if self._find_by_name(factor_name):
            print(f"  ⚠️ 因子 '{factor_name}' 已存在，跳过")
            return False

        # 相似定义检查
        similar = self._find_similar(factor.get("definition", ""))
        if similar and not auto_yes and input(f"  发现相似因子 '{similar['factor_name']}'，仍要添加?(y/n): ").strip().lower() != 'y':
            return False

        # 填充元数据
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        factor["added_at"] = now
        factor["updated_at"] = now
        factor["status"] = factor.get("status", "pending")
        factor["modality"] = factor.get("modality", "single")
        factor["tags"] = factor.get("tags", [])
        factor["quality_score"] = factor.get("quality_score", 0)
        factor["backtest_result"] = factor.get("backtest_result", {})
        factor["notes"] = factor.get("notes", "")

        self.factors.append(factor)
        print(f"  ✅ 已添加: {factor_name}")
        return True

    def add_factors(self, factors: List[Dict], auto_yes: bool = False) -> int:
        """批量添加因子（auto_yes=True 跳过交互确认）"""
        added = 0
        for factor in factors:
            if self.add_factor(factor, auto_yes=auto_yes):
                added += 1
        if added > 0:
            self.save()
        return added

    # ------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------

    def update_status(self, factor_name: str, status: str, notes: str = "") -> bool:
        """
        更新因子状态

        Args:
            factor_name: 因子名称
            status: pending / approved / rejected / active / deprecated
            notes: 批注（可选）
        """
        for factor in self.factors:
            if factor.get("factor_name") == factor_name:
                old_status = factor.get("status", "")
                factor["status"] = status
                factor["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if status == "approved":
                    factor["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if notes:
                    factor["notes"] = notes

                print(f"  ✅ {factor_name}: {old_status} → {status}")
                return True

        print(f"  ⚠️ 未找到因子: {factor_name}")
        return False

    def approve_factor(self, factor_name: str, notes: str = "") -> bool:
        """批准因子（进入待回测状态）"""
        return self.update_status(factor_name, "approved", notes)

    def reject_factor(self, factor_name: str, notes: str = "") -> bool:
        """拒绝因子"""
        return self.update_status(factor_name, "rejected", notes)

    def activate_factor(self, factor_name: str, notes: str = "") -> bool:
        """激活因子（进入实盘池）"""
        return self.update_status(factor_name, "active", notes)

    def batch_update_status(
        self,
        factor_names: List[str],
        status: str,
        notes: str = "",
    ) -> int:
        """批量更新状态"""
        count = 0
        for name in factor_names:
            if self.update_status(name, status, notes):
                count += 1
        if count > 0:
            self.save()
        return count

    # ------------------------------------------------------
    # 查询
    # ------------------------------------------------------

    def get_by_status(self, status: str) -> List[Dict]:
        """获取指定状态的因子"""
        return [f for f in self.factors if f.get("status") == status]

    def get_pending(self) -> List[Dict]:
        """获取待审因子"""
        return self.get_by_status("pending")

    def get_approved(self) -> List[Dict]:
        """获取已批准因子"""
        return self.get_by_status("approved")

    def get_active(self) -> List[Dict]:
        """获取活跃（实盘）因子"""
        return self.get_by_status("active")

    def get_multimodal(self) -> List[Dict]:
        """获取多模态因子"""
        return [f for f in self.factors if f.get("modality") == "multimodal"]

    def search(self, keyword: str) -> List[Dict]:
        """全文搜索因子"""
        results = []
        kw = keyword.lower()
        for f in self.factors:
            text = " ".join([
                f.get("factor_name", ""),
                f.get("definition", ""),
                f.get("paper_title", ""),
                " ".join(f.get("tags", [])),
            ]).lower()
            if kw in text:
                results.append(f)
        return results

    def get_by_tag(self, tag: str) -> List[Dict]:
        """获取指定标签的因子"""
        return [f for f in self.factors if tag in f.get("tags", [])]

    # ------------------------------------------------------
    # 统计分析
    # ------------------------------------------------------

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = len(self.factors)
        status_counts = {}
        for s in ["pending", "approved", "rejected", "active", "deprecated"]:
            status_counts[s] = len([f for f in self.factors if f.get("status") == s])

        conf_counts = {"高": 0, "中": 0, "低": 0}
        for f in self.factors:
            c = f.get("confidence", "")
            if c in conf_counts:
                conf_counts[c] += 1

        modality_counts = {
            "single": len([f for f in self.factors if f.get("modality") != "multimodal"]),
            "multimodal": len([f for f in self.factors if f.get("modality") == "multimodal"]),
        }

        tagged = len([f for f in self.factors if f.get("tags")])
        with_backtest = len([f for f in self.factors if f.get("backtest_result")])

        return {
            "total": total,
            "status": status_counts,
            "confidence": conf_counts,
            "modality": modality_counts,
            "tagged": tagged,
            "with_backtest": with_backtest,
        }

    def get_grouped_by_source(self) -> Dict[str, List[Dict]]:
        """按论文来源分组"""
        groups = {}
        for f in self.factors:
            source = f.get("source", f.get("paper_title", "未知"))
            if source not in groups:
                groups[source] = []
            groups[source].append(f)
        return groups

    # ------------------------------------------------------
    # 导出
    # ------------------------------------------------------

    def export_to_markdown(self, output_path: Optional[str] = None):
        """导出为Markdown报告"""
        if output_path is None:
            from config import OUTPUT_DIR
            output_path = OUTPUT_DIR / f"factor_report_{datetime.now().strftime('%Y%m%d')}.md"

        lines = [
            "# 因子候选库报告\n",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"总计: {len(self.factors)} 个因子\n",
        ]

        stats = self.get_statistics()
        lines.append("## 统计概览\n")
        lines.append(f"- 总因子数: {stats['total']}")
        lines.append(f"- 待审: {stats['status'].get('pending', 0)}")
        lines.append(f"- 已批准: {stats['status'].get('approved', 0)}")
        lines.append(f"- 活跃: {stats['status'].get('active', 0)}")
        lines.append(f"- 已拒绝: {stats['status'].get('rejected', 0)}")
        lines.append(f"- 多模态因子: {stats['modality'].get('multimodal', 0)}\n")

        lines.append("## 因子列表（按状态分组）\n")
        for status in ["pending", "approved", "active", "rejected"]:
            group = self.get_by_status(status)
            if not group:
                continue
            lines.append(f"### {status.upper()} ({len(group)}个)\n")
            for f in group:
                name = f.get("factor_name", "未知")
                modality = f.get("modality", "single")
                modality_icon = "🔗" if modality == "multimodal" else "📊"
                lines.append(f"#### {modality_icon} {name}\n")
                lines.append(f"- 状态: `{f.get('status', 'unknown')}`")
                lines.append(f"- 置信度: {f.get('confidence', 'unknown')}")
                lines.append(f"- 定义: {f.get('definition', 'N/A')}")
                lines.append(f"- 数据需求: {f.get('data_needed', 'N/A')}")
                lines.append(f"- 适用市场: {f.get('market_scope', 'unknown')}")

                if f.get("modality") == "multimodal":
                    lines.append(f"- 融合方法: {f.get('fusion_method', 'N/A')}")
                    lines.append(f"- 新闻贡献: {f.get('news_contribution', 'N/A')}")

                lines.append(f"- 论文: [{f.get('paper_title', 'N/A')}]({f.get('paper_link', '')})")
                lines.append(f"- 论文结论: {f.get('paper_conclusion', 'N/A')}")

                if f.get("backtest_result"):
                    br = f["backtest_result"]
                    lines.append(f"- **回测结果**: IC={br.get('ic','?')}, ICIR={br.get('icir','?')}, TOP={br.get('top_return','?')}%")

                lines.append(f"- 添加时间: {f.get('added_at', 'unknown')}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"已导出报告到 {output_path}")

    def export_to_csv(self, output_path: Optional[str] = None):
        """导出为CSV（便于Excel分析）"""
        if output_path is None:
            from config import OUTPUT_DIR
            output_path = OUTPUT_DIR / f"factors_{datetime.now().strftime('%Y%m%d')}.csv"

        import csv
        fieldnames = [
            "factor_name", "status", "modality", "confidence",
            "definition", "data_needed", "market_scope",
            "fusion_method", "news_contribution", "news_required",
            "source", "paper_title", "paper_link",
            "quality_score", "added_at", "updated_at",
        ]

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for f in self.factors:
                row = {k: f.get(k, "") for k in fieldnames}
                writer.writerow(row)

        print(f"已导出CSV到 {output_path}")

    # ------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------

    def _find_by_name(self, name: str) -> Optional[Dict]:
        """按名称查找因子"""
        for f in self.factors:
            if f.get("factor_name") == name:
                return f
        return None

    def _find_similar(self, definition: str, threshold: float = 0.8) -> Optional[Dict]:
        """查找定义相似的因子（简化版：关键词重叠）"""
        if not definition:
            return None

        words = set(re.findall(r"\w{3,}", definition.lower()))

        for f in self.factors:
            exist_words = set(re.findall(r"\w{3,}", f.get("definition", "").lower()))
            if not exist_words:
                continue

            overlap = len(words & exist_words) / max(len(words), 1)
            if overlap > threshold:
                return f

        return None


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    db = FactorDatabase()
    print("\n统计信息:", db.get_statistics())
    print("\n待审因子:", len(db.get_pending()))
    print("多模态因子:", len(db.get_multimodal()))
