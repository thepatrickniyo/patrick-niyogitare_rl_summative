# Final report (7–10 pages PDF) — CodetyAI RL

Use your course template. Keep sections concise.

1. **Title & abstract** — Adaptive learning & mentorship optimization for CodetyAI; link to reducing STEM youth unemployment in Rwanda.
2. **Introduction** — Academic–industry gap; centralized training + mentorship; why RL for sequencing interventions.
3. **Environment**
   - **State:** skill, engagement (low/med/high), projects, mentorship sessions, confidence, time progress (normalized observation in `custom_env.py`).
   - **Actions:** beginner lesson, advanced lesson, project, mentor, revision.
   - **Rewards:** +10 / +15 / +20 / −10 / −20 as implemented; tie to employability.
   - **Start / terminal:** random novice profile; success (job-ready), failure (dropout), truncation.
4. **Diagram** — Reuse README mermaid; caption: observations → policy → platform actions → student outcomes.
5. **Algorithms** — DQN, REINFORCE, PPO, A2C; key tuned hyperparameters (`hyperparam_runs.py`).
6. **Experiments** — Same env, evaluation protocol, 10 runs × 4 algorithms; hardware/runtime.
7. **Results** — Table of best `eval_mean_return`; optional TensorBoard curves.
8. **Discussion** — Which method stabilized; exploration vs variance; ethical note (real students ≠ simulation).
9. **Deployment** — How a production CodetyAI could use RL-assisted scheduling with human oversight.
10. **Conclusion & references**

**Video checklist:** problem (mission), agent behaviour, rewards, objective, best agent run with **GUI + verbose terminal**.
