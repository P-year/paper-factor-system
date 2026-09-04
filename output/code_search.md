# 代码搜索报告：2510.15691

**论文：** Exploring the Synergy of Quantitative Factors and Newsflow Representations from Large Language Models for Stock Return Prediction
**arXiv:** [2510.15691](https://arxiv.org/abs/2510.15691)

---

## 1. 搜索记录

### 1.1 GitHub搜索
- **关键词:** `2510.15691`, `Tian Guo stock return prediction`, `quantitative factors newsflow LLM`
- **结果:** ❌ 未找到对应仓库
- **原因:** 论文较新（2025-10），作者单位为RAM Active Investments（商业机构），大概率不开源

### 1.2 Papers With Code
- **URL:** https://paperswithcode.com/paper/exploring-the-synergy-of-quantitative-factors
- **结果:** ❌ 页面无法访问或未收录

### 1.3 Semantic Scholar
- **API查询:** `/graph/v1/paper/arXiv:2510.15691`
- **结果:** 找到paperId = `88f47a944ab8b957c86393b0900b3d93161473cd`，但返回429（请求过多）

### 1.4 Google/Bing搜索
- **搜索词:** `site:github.com "quantitative factors" "newsflow" "stock return" "Tian Guo"`
- **结果:** ❌ 未找到相关代码

### 1.5 作者信息
- **Tian Guo, Emmanuel Hauptmann**
- **机构:** RAM Active Investments, Geneva, Switzerland
- **性质:** 商业投资公司，非学术机构，开源可能性较低

---

## 2. 相关可参考开源项目

虽然没有直接对应代码，但以下项目提供了可参考的实现模块：

### 2.1 类似多模态融合方法
| 项目 | 语言 | 说明 |
|------|------|------|
| [DeBERTa-v3 fine-tuning](https://github.com/microsoft/DeBERTa) | Python | 论文使用的LLM基座，微调参考 |
| [LoRA-PEFT](https://github.com/huggingface/peft) | Python | LoRA参数高效微调实现 |
| [FININ (Hu et al.)](https://github.com/) | Python | 新闻+因子融合baseline参考 |

### 2.2 金融文本Embedding
| 项目 | 语言 | 说明 |
|------|------|------|
| [Jina AI Embeddings](https://github.com/jina-ai/jina-embeddings) | Python | 多语言embedding，可替代DeBERTa |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Python | 金融LLM微调框架 |
| [AugCapital/FINMEM](https://github.com/AugCapital/FINMEM) | Python | 金融记忆增强LLM |

### 2.3 量化因子+机器学习
| 项目 | 语言 | 说明 |
|------|------|------|
| [Japanese Stock Selection DS](https://github.com/yotamagra/japanese-stock-selection-ds) | Python | 因子+ML选股，可作框架参考 |
| [QLIB](https://github.commicrosoft/qlib) | Python | 量化研究平台，含因子库和回测 |
| [AlphaPool](https://github.com/AlphaPoolProject/AlphaPool) | Python | 多因子库 |

---

## 3. 论文实现的关键模块（需自行实现）

### 3.1 必须实现的模块

```python
# 1. 因子 + 新闻融合网络
class FusionCombination(nn.Module):
    def __init__(self, factor_dim, news_dim, hidden_dim=128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(factor_dim + news_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x_factors, x_news):
        x = torch.cat([x_factors, x_news], dim=-1)
        return self.fusion(x)

# 2. Mixture Model with Decoupled Training
class MixtureDecoupled(nn.Module):
    def __init__(self, factor_net, fusion_net, logits_net):
        self.factor_net = factor_net    # 纯因子预测
        self.fusion_net = fusion_net    # 融合预测
        self.logits_net = logits_net     # 混合权重网络
    
    def forward(self, x_f, x_n):
        r_f = self.factor_net(x_f)
        x_u = torch.cat([x_f, x_n], dim=-1)
        r_u = self.fusion_net(x_u)
        probs = F.softmax(self.logits_net(torch.cat([x_f, x_n], dim=-1)), dim=-1)
        return probs[:, 0] * r_f + probs[:, 1] * r_u

# 3. Decoupled Training Step
def decoupled_train_step(model, optimizer, x_f, x_n, r, tau=1.0):
    # Stage 1: 独立训练各组件
    optimizer.zero_grad()
    loss_indep = mse(model.factor_net(x_f), r) + mse(model.fusion_net(x_u), r)
    loss_indep.backward()
    optimizer.step()
    
    # Stage 2: 分布匹配（固定θ_f, θ_u更新φ）
    with torch.no_grad():
        r_f_fixed = model.factor_net(x_f)
        r_u_fixed = model.fusion_net(x_u)
        target_probs = F.softmax(-(r - r_f_fixed)**2 / tau, dim=-1)
    
    optimizer.zero_grad()
    current_probs = model.logits_net(torch.cat([x_f, x_n], dim=-1))
    loss_dist = F.kl_div(current_probs, target_probs, reduction='batchmean')
    loss_dist.backward()
    optimizer.step()
```

---

## 4. 总结

**开源状态：** ❌ 论文代码未公开（商业机构）
**替代方案：** 参照上述开源项目和论文描述的架构自行实现
**推荐路径：** 以QLIB为研究平台，叠加FinBERT/jina-embeddings做新闻表示，实现论文核心架构
