# Sampling and Aliasing for Vision Models

## 1. 목적
sampling theorem, aliasing, anti-aliasing을  
resize/downsample/pooling/stride 설계와 직접 연결한다.

## 2. 수식 중심 요약
- Nyquist 조건: `f_s >= 2 f_max`
- 샘플링: `x[n] = x(nT_s)`
- aliasing: `f > f_s/2` 성분이 낮은 주파수로 fold
- downsample before low-pass = aliasing 위험 증가

## 3. 직관 중심 요약
- 해상도를 줄이는 것은 "정보를 버리는 것"
- 버리기 전에 고주파를 정리하지 않으면 가짜 패턴이 생김
- aliasing은 학습 데이터 품질과 모델 불변성에 직접 악영향

## 4. 비전 연결

### 4.1 이미지 resize
- bilinear/bicubic/area 보간 방식이 aliasing 특성을 다르게 만든다.
- 학습/추론 resize 파이프라인 불일치가 성능 차이를 만든다.

### 4.2 CNN downsampling
- stride conv, max-pool, avg-pool은 모두 샘플링 연산 포함
- anti-aliasing filter를 선행하면 shift robustness 개선 가능

### 4.3 비디오 모델
- temporal sampling(frame skip)에서도 aliasing 발생
- 빠른 motion에서 action recognition 성능 저하 원인

## 5. 논문에서 보이는 형태
- "anti-aliased CNN"
- "blur pooling"
- "shift invariance / consistency"
- "multi-rate feature processing"

## 6. 구현 체크리스트
1. downsample 전 low-pass 여부 명시
2. train/eval resize 방식 통일
3. augmentation이 aliasing을 과도하게 만들지 검증
4. frequency-domain metric으로 artifact 점검
5. video는 spatial + temporal sampling 모두 관리

## 7. 미니 실험 아이디어
```python
import torch
import torch.nn.functional as F

x = torch.randn(1,1,256,256)

# naive downsample
y0 = x[..., ::2, ::2]

# anti-aliased downsample
k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
k = (k/k.sum()).view(1,1,3,3)
x_lp = F.conv2d(x, k, padding=1)
y1 = x_lp[..., ::2, ::2]

print("naive std:", float(y0.std()), "anti-aliased std:", float(y1.std()))
```

## 8. 자주 틀리는 포인트
1. "고해상도면 aliasing 걱정 없음" -> downsample 단계에서 다시 발생
2. "pooling은 안정적" -> anti-aliasing 없는 pooling도 artifact 유발 가능
3. "train에서 랜덤 리사이즈 했으니 괜찮다" -> eval 파이프라인이 다르면 문제 재발

## 9. 다음 문서
- [fourier](./fourier.md)
- [vision-connections](./vision-connections.md)
- [signals-basics](./signals-basics.md)
