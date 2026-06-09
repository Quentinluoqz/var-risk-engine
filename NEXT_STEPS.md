## Next Steps — 项目演进路线图

当前版本已完成 P0/P1/P2/P3 的核心修复：命令入口可用、完整离线 demo、YAML 配置、结构化输出、GARCH-MC 接入、Basel 口径收紧、t-Copula likelihood 修正、EVT 阈值诊断、CI、模型验证文档、FRTB 说明、模型 inventory、项目级缓存、离线样例数据、Docker 和 pre-commit。

---

### P0 — 投递前必须修复项 ✅ 已完成

**1. CLI 与可复现运行** ✅ DONE
- 新增 `src/var_risk_engine/cli.py`
- `pyproject.toml` 的 `var-engine` 入口已指向可用 wrapper
- 支持 `--tickers`、`--weights`、`--start`、`--end`、`--confidence`、`--n-sims`
- 支持 `--input data/sample_prices.csv`
- 支持 `--offline`
- 支持 `--config configs/demo.yaml`
- 支持 `--output outputs/<run_name>`
- `var-engine --help` 只显示帮助，不再误跑完整分析

**2. Basel traffic-light 口径修正** ✅ DONE
- Kupiec / Christoffersen 继续用于一般 VaR backtesting
- Basel traffic-light 仅在 99% VaR / 250 observation 标准口径下应用
- 其他设置下输出 “not applied”，避免把诊断回测误包装成监管结论

**3. GARCH-adjusted Monte Carlo** ✅ DONE
- Baseline MC 仍使用历史年化波动率
- 新增 GARCH-MC：用各资产一日 GARCH 波动率预测替代常数 sigma
- VaR/ES 汇总表新增 `GARCH-MC`

**4. 工程质量门禁** ✅ DONE
- `ruff check src tests` 已通过
- 测试增至 87 个
- 移除风险归因函数中的控制台打印副作用

---

### P1 — 模型可信度增强 ✅ 已完成

**5. EVT 阈值诊断** ✅ DONE
- 新增 `mean_excess_table`
- 新增 `threshold_stability_table`
- 主流程输出 EVT threshold diagnostics
- 新增图：`reports/figures/evt_threshold_diagnostics.png`

**6. t-Copula likelihood 修正** ✅ DONE
- t-Copula likelihood 改为使用 pseudo-observations 的 Student-t inverse CDF
- 修正 `rho <= -1` 的 tail dependence 边界处理
- 保留 Gaussian vs t-Copula tail dependence 对比

**7. README 真实性修复** ✅ DONE
- README 已更新为当前代码真实能力
- 明确说明 Basel 适用条件、GARCH-MC、EVT 诊断、6 个 notebooks、87 tests、offline demo、YAML config、outputs
- 删除过度包装的 “regulatory-grade” 式表达

---

### P2 — 工程化与投递材料 ✅ 已完成

**8. CI/CD 与覆盖率** ✅ DONE
- 新增 `.github/workflows/ci.yml`
- Python 3.10 / 3.12 matrix
- CI 执行 `ruff check src tests`
- CI 执行 `pytest --cov=var_risk_engine --cov-report=term-missing`
- 本地 coverage 命令通过，总覆盖率当前约 90%

**9. 模型验证文档** ✅ DONE
- 新增 `docs/model_validation_note.md`
- 新增 `docs/model_inventory.md`
- 覆盖 model purpose、data、method inventory、backtesting evidence、EVT validation、limitations、challenger models
- 对 BNP Paribas / SocGen / Natixis 的 Model Risk 或 Market Risk 面试价值较高

**10. FRTB/Basel III 说明** ✅ DONE
- 新增 `docs/frtb_discussion.md`
- 解释 VaR 到 ES 的监管迁移、Basel traffic-light 适用口径、FRTB 缺口
- 明确本项目是 analytical prototype，不是完整 capital engine

**11. 数据缓存与离线样例** ✅ DONE
- 默认缓存迁移到项目级 `data/cache/`
- 支持 `VAR_RISK_ENGINE_CACHE_DIR` 环境变量覆盖
- 保留旧 `src/var_risk_engine/data/` cache fallback
- 新增 `data/sample_prices.csv`

**12. 结构化输出** ✅ DONE
- 每次 run 输出 `metrics.json`
- 输出 `var_es_table.csv`
- 输出 `backtesting_results.csv`
- 输出 `stress_results.csv`
- 输出 `risk_attribution.csv`
- 输出 `data_quality.csv`
- 输出 `report.html`
- 图表输出至 `<output>/figures/`

**13. 数据质量模块** ✅ DONE
- 新增 `src/var_risk_engine/data_validation.py`
- 检查 duplicate date、missing price、non-positive price、stale price、return outlier、zero volatility

**14. Docker 与 pre-commit** ✅ DONE
- 新增 `Dockerfile`
- 新增 `.pre-commit-config.yaml`
- Makefile 新增 `docker-build` / `docker-run`

---

### P4 — 后续增强建议

**15. 更深层结构拆分**
- `main.py` 的 section-level plotting 已迁移到 `reporting/plots.py`
- 将 table builders 迁移到 `reporting/tables.py`
- 将组合级计算迁移到 `engine/portfolio_engine.py`
- 将 rolling forecast 迁移到 `engine/backtest_engine.py`

**16. 性能和生产化**
- Monte Carlo 支持完整路径输出，不只 terminal distribution
- 可选 Numba 加速
- 增加 coverage badge
- 增加 Filtered Historical Simulation
- 增加 Student-t innovations / bootstrap Monte Carlo

---

### 简历描述建议

> Built a modular Python market-risk engine estimating Historical, Parametric, Monte Carlo, and GARCH-adjusted Monte Carlo VaR/Expected Shortfall for a multi-asset portfolio. Added covariance estimation, GARCH-family volatility models, stress testing, EVT Peaks-over-Threshold tail modelling with threshold diagnostics, Gaussian vs Student-t copula tail-dependence analysis, Euler VaR attribution, and VaR backtesting via Kupiec and Christoffersen tests. Packaged with a one-command offline demo, YAML configuration, structured JSON/CSV/HTML outputs, 87 tests, ~90% coverage, GitHub Actions CI, Docker, pre-commit, Jupyter notebooks, and model-validation documentation.

这个版本可以投递，但面试时建议说成 “market-risk and model-validation research engine”，不要说成完整监管资本系统。
