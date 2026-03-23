# Vision Connections: Signals and Systems in Practice

## 1. 목적
신호및시스템 개념을 실제 비전 태스크(분류/검출/분할/복원/생성)에 매핑한다.

## 2. 태스크별 연결

### 2.1 분류/검출/분할
- 다운샘플링/feature pyramid에서 aliasing 제어가 localization 성능에 영향
- backbone의 frequency bias가 작은 객체 인식에 불리할 수 있음
- anti-aliased downsampling이 shift consistency를 높일 수 있음

### 2.2 복원 (denoising/deblurring/super-resolution)
- forward model: `y = h*x + n`
- denoising: noise 모델(gaussian/poisson/read noise) 명확화가 핵심
- deblurring: blur kernel 추정 + 역문제 정규화
- super-resolution: 고주파 hallucination과 실제 복원 구분 필요

### 2.3 생성 모델
- diffusion은 점진적 noise injection/removal을 시스템 관점으로 해석 가능
- 주파수 대역별 복원 난이도가 다르므로 loss/architecture의 주파수 편향 분석 필요

## 3. CNN/ViT 연결
- CNN: 로컬 LTI 근사 + 비선형 활성화
- ViT: patchify가 하나의 샘플링 연산이며, patch size가 aliasing/세부정보에 영향
- hybrid 모델: convolution 전처리 + attention 결합으로 주파수 특성을 조절

## 4. noise and denoising intuition
- 수식: `y = x + n`, 또는 `y = h*x + n`
- 직관: 신호와 잡음을 분리하는 것은 prior 설계 문제
- 실무:
  - low-light 데이터에서 shot noise 지배
  - 단순 low-pass는 디테일 훼손
  - 학습형 denoiser는 데이터 분포 의존성이 큼

## 5. filtering 전략
- pre-filtering: aliasing/노이즈 완화
- mid-filtering: feature 안정화
- post-filtering: artifact 억제
- 핵심: 필터링 단계마다 손실되는 정보와 얻는 안정성을 명시적으로 trade-off

## 6. 논문 읽기 체크리스트
1. 신호 모델 가정이 현실 데이터와 맞는가?
2. sampling/resize/augment가 명확히 기술돼 있는가?
3. frequency 분석이 시각화로만 끝나지 않고 성능과 연결되는가?
4. denoising/deblurring에서 noise/blur 통계가 보고되는가?
5. backbone 변경보다 preprocessing 변경이 성능에 더 큰 영향은 없는가?

## 7. 구현 체크리스트
1. 입력 전처리(리사이즈/보간/정규화) 버전 고정
2. 학습/추론 파이프라인 일치 검증
3. 복원 task는 PSNR/SSIM + perceptual metric 동시 기록
4. frequency-domain 진단(FFT magnitude) 자동 저장
5. 소규모 sanity set으로 artifact를 눈으로 확인

## 8. 미니 실험 아이디어
```python
import torch
import torch.nn.functional as F

# noise + denoise toy
x = torch.randn(1,1,128,128)
noisy = x + 0.1 * torch.randn_like(x)

k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
k = (k/k.sum()).view(1,1,3,3)
den = F.conv2d(noisy, k, padding=1)

print("noisy mse:", float(((x-noisy)**2).mean()))
print("den mse:", float(((x-den)**2).mean()))
```

## 9. 연구 메모 템플릿 제안
- 문제 정의: 신호 관점에서 무엇이 손상되는가?
- 시스템 모델: `y = H{x} + n`에서 `H,n` 가정은?
- 주파수 분석: 어떤 대역이 손실/왜곡되는가?
- 대응 전략: filtering, sampling, loss, architecture 중 무엇으로 해결할 것인가?
- 검증: 공간+주파수+다운스트림 지표를 모두 확인했는가?

## 10. 연결 문서
- [signals-basics](./signals-basics.md)
- [convolution-lti](./convolution-lti.md)
- [fourier](./fourier.md)
- [sampling](./sampling.md)
