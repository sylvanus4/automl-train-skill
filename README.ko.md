# automl-train

> 언어: [English](README.md) · 한국어

어떤 학습 백엔드와도 동작하는 **스키마 게이트 기반 하이퍼파라미터 탐색(HPO/AutoML) 에이전트
스킬**입니다. 단일 학습 실행을 규율 있는 탐색으로 바꾸고, 무엇보다 **이기지 못한 결과를 승격하지
않습니다**. 이식 가능한 `SKILL.md` 계약 + 표준 라이브러리만 쓰는 8개 스크립트 + GPU 없이 도는
end-to-end 예제로 구성됩니다.

## 왜

대부분의 AutoML 데모는 끝에 좋은 숫자만 보여줍니다. 정작 흥미로운 질문은 탐색이 개선을 **못 찾았을
때** 무슨 일이 일어나느냐입니다. 이 스킬은 그걸 정직하게 답합니다. 다섯 개의 게이트를 결정론 코드가
소유해서, 실행을 시작할 수 있는지와 결과를 승격할 수 있는지를 결정합니다. 모델은 다음 하이퍼파라미터만
추천하고 나머지는 코드가 합니다.

게이트 철학은 NVIDIA의 [`tao-run-automl`](https://github.com/NVIDIA-TAO/tao-skill-bank)
스킬(Apache-2.0)에서 빌렸습니다. TAO는 GPU를 쓰기 전에 추천값·지표·탐색 공간·예상 실행 시간을
검토합니다. 다른 점은 특정 trainer에 묶이지 않고(백엔드는 직접 연결), 게이트가 1분이면 읽는 순수
Python이라는 것입니다.

## 다섯 개의 게이트

1. **스키마 없으면 탐색 없음** — 잘못됐거나 범위가 없거나 학습 스텝 0인 탐색 공간은 루프를 시작하지 못합니다.
2. **튜닝 전 baseline** — 기준 지표를 먼저 기록해야 합니다.
3. **실행 전 launch review** — `max_concurrent × gpus_per_trial ≤ gpu_cap` + 예상 wall-time이 한도를 넘으면 막습니다.
4. **독립 최종 평가** — 우승 trial이 같은 평가셋에서 baseline을 이겨야 승격합니다.
5. **예산 태우지 않고 실패** — 실패를 data/image-cred/infra/spec-schema/model-code로 분류하고, systemic 원인이 반복되면 예산 소진 전에 멈춥니다.

탐색 전략: **random**, **ASHA**(successive halving), **Hyperband**(bracketed successive halving).

## 빠른 시작 / 검증 (GPU 0)

```bash
python tests/smoke.py              # 9개 체크: 게이트가 게이트하는지, 실패 분류, 루프 조립
python examples/run_local_sweep.py # 전 과정: baseline → 6 trial → 선정 → 최종 게이트 → 리포트
```

`run_local_sweep.py`가 곧 워크드 예제입니다. 이걸 복사해서 `run_trial()`만 여러분의
`submit()` + `scrape()`로 바꾸면 실제 trainer에 연결됩니다.

**실패 정책**: systemic 원인(data/image-cred/infra/spec-schema)만 예산보존 조기중단을 유발합니다.
새 trial도 똑같이 실패할 것이기 때문입니다. model-code 실패(예: 발산하는 학습률)는 config별 문제라
탐색을 멈추지 않고 계속합니다.

## 백엔드 연결 (직접 쓰는 부분)

스크립트는 평평한 key/value trial 설정과 숫자 지표만 주고받습니다. 어댑터 두 개를 제공하면 됩니다.

| 어댑터 | 역할 | 예시 |
|---|---|---|
| `submit(config) -> run_id` | 학습 1회 실행 | Kubeflow Trainer v2 `TrainJob`, Ray Tune, SLURM `sbatch`, 로컬 `subprocess` |
| `scrape(run_id) -> value` | 목표 지표 회수 | `mlflow_scrape.py`(레퍼런스), W&B, stdout 파싱 |

스키마의 `tunable_env`가 탐색 가능한 키를 선언하므로, 검증기가 특정 trainer의 파라미터 이름에
묶이지 않습니다.

## 크레딧 / 라이선스

- 게이트 규율은 NVIDIA [`tao-run-automl`](https://github.com/NVIDIA-TAO/tao-skill-bank)(Apache-2.0)에서 차용. [`NVIDIA/skills`](https://github.com/NVIDIA/skills)도 참고.
- Apache-2.0.
