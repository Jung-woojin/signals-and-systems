# 이미지/컴퓨터비전/딥러닝 연구자를 위한 신호및시스템 실전 가이드

## 1. 이 문서의 목적
이 문서는 전자공학 교재 요약이 아니라, 비전 연구자가 논문을 읽고 모델을 설계할 때 바로 참고할 수 있는 신호및시스템 관점을 제공한다.  
핵심 목표는 다음 3가지다.
- 이미지와 CNN 연산을 "시스템 관점"으로 해석하기
- 주파수/샘플링/필터링 개념을 성능 문제 디버깅에 연결하기
- 수식과 직관을 함께 사용해 연구 의사결정 속도를 높이기

## 2. 왜 비전 연구자에게 신호및시스템이 중요한가
- 이미지 자체가 2D 이산 신호다.
- CNN의 핵심 연산(convolution)은 LTI 시스템의 연산과 직접 연결된다.
- aliasing, blur, noise, sampling artifact는 모델 성능 저하의 주요 원인이다.
- frequency bias(저주파 선호) 같은 현상은 신호 관점 없이 해석이 어렵다.
- anti-aliasing, denoising, deblurring, Fourier feature 등 최신 비전 기법도 결국 신호 처리 개념을 활용한다.

## 3. 핵심 개념 지도
1. signals and systems overview  
2. discrete-time vs continuous-time intuition  
3. LTI systems  
4. impulse response  
5. convolution  
6. frequency response  
7. Fourier series / Fourier transform  
8. discrete Fourier transform intuition  
9. Laplace transform overview  
10. Z-transform overview  
11. poles and zeros  
12. sampling theorem  
13. aliasing  
14. filtering and filter design  
15. modulation overview  
16. noise and denoising intuition

## 4. 개념별 설명

### 4.1 signals and systems overview
- 수식 중심 요약: 입력 `x`가 시스템 `H`를 거쳐 출력 `y=H{x}`를 만든다.
- 직관 중심 요약: 시스템은 "이미지에 어떤 변형을 가하는 연산 규칙"이다.
- 비전 연결: 카메라/렌즈/센서/ISP/CNN 모두 연쇄 시스템으로 볼 수 있다.

### 4.2 discrete-time vs continuous-time intuition
- 수식 중심 요약: 연속 신호 `x(t)`를 샘플링하면 이산 신호 `x[n]=x(nT_s)`를 얻는다.
- 직관 중심 요약: 현실 세계는 연속이지만 디지털 이미지/영상은 격자 샘플이다.
- 비전 연결: frame rate, spatial resolution, temporal sampling이 인식/추적 성능에 직접 영향.

### 4.3 LTI systems
- 수식 중심 요약: 선형성 + 시간(공간) 불변성 만족 시 `y = x * h`.
- 직관 중심 요약: 입력을 합치면 출력도 합쳐지고, 위치를 옮기면 출력도 같이 옮겨진다.
- 비전 연결: 많은 초기 비전 필터(blur, sharpen, edge)는 LTI 근사로 이해 가능.

### 4.4 impulse response
- 수식 중심 요약: impulse 입력 `δ`에 대한 출력 `h`가 시스템을 완전히 규정.
- 직관 중심 요약: 시스템의 "지문" 또는 "커널 프로파일".
- 비전 연결: blur kernel 추정(deblurring), PSF(point spread function) 모델링.

### 4.5 convolution
- 수식 중심 요약: `y[n]=Σ_k x[k] h[n-k]` (2D는 공간 좌표로 확장).
- 직관 중심 요약: 주변 픽셀을 커널 가중치로 합성해 새로운 픽셀 생성.
- 비전 연결: CNN conv, smoothing, edge detection, texture extraction.

### 4.6 frequency response
- 수식 중심 요약: `H(ω)`는 각 주파수 성분의 증폭/감쇠/위상 이동을 나타냄.
- 직관 중심 요약: "어떤 무늬(주파수)를 통과시키고 무엇을 막는가".
- 비전 연결: 저주파는 조명/큰 구조, 고주파는 엣지/디테일.

### 4.7 Fourier series / Fourier transform
- 수식 중심 요약: 신호를 주파수 기저의 합으로 분해.
- 직관 중심 요약: 복잡한 이미지 패턴을 "주파수 성분 조합"으로 본다.
- 비전 연결: Fourier-based image analysis, 주파수 도메인 augmentation, texture 분석.

### 4.8 discrete Fourier transform intuition
- 수식 중심 요약: DFT는 유한 길이 이산 신호의 주파수 좌표 변환.
- 직관 중심 요약: 픽셀 공간을 주파수 격자로 재표현.
- 비전 연결: FFT 기반 convolution 가속, 주파수 마스킹, artifact 분석.

### 4.9 Laplace transform overview
- 수식 중심 요약: `X(s)=∫ x(t)e^{-st}dt`, 연속 시스템의 안정성/응답 해석 도구.
- 직관 중심 요약: 시간 함수 해석을 복소 평면으로 옮겨 시스템 성질을 보기 쉽게 만든다.
- 비전 연결: 카메라 모션 blur를 연속 시스템으로 근사할 때 이론적 프레임 제공.

### 4.10 Z-transform overview
- 수식 중심 요약: `X(z)=Σ x[n] z^{-n}`, 이산 시스템의 전달함수 분석 도구.
- 직관 중심 요약: 디지털 필터/CNN-like 이산 연산의 안정성과 주파수 특성을 한 번에 본다.
- 비전 연결: 영상 필터 설계, 재귀형 denoising filter 해석.

### 4.11 poles and zeros
- 수식 중심 요약: 전달함수 `H(z)=B(z)/A(z)`에서 zero는 출력 억제 주파수, pole은 시스템 응답 증폭 특성.
- 직관 중심 요약: 시스템의 "강조점"과 "소거점".
- 비전 연결: 필터 설계에서 ringing/overshoot/안정성 문제 진단.

### 4.12 sampling theorem
- 수식 중심 요약: 샘플링 주파수는 최대 주파수의 2배 이상(`f_s >= 2f_max`)이어야 aliasing 회피.
- 직관 중심 요약: 너무 촘촘하지 않게 찍으면 패턴이 왜곡되어 다른 패턴처럼 보임.
- 비전 연결: 다운샘플링, stride conv, pooling 설계의 이론적 기준.

### 4.13 aliasing
- 수식 중심 요약: 고주파 성분이 낮은 주파수로 fold되어 왜곡 발생.
- 직관 중심 요약: 가짜 무늬/깜빡임/모아레가 생김.
- 비전 연결: anti-aliased CNN, resize artifact, video temporal aliasing.

### 4.14 filtering and filter design
- 수식 중심 요약: low-pass/high-pass/band-pass/notch 설계로 신호 대역 제어.
- 직관 중심 요약: 필요한 정보만 통과시키고 방해 성분을 억제.
- 비전 연결: smoothing, edge enhancement, denoising, pre-processing.

### 4.15 modulation overview
- 수식 중심 요약: 신호를 다른 주파수 대역으로 이동시키는 변조(곱셈/주파수 이동).
- 직관 중심 요약: 정보를 다른 "채널"로 옮겨 표현하기.
- 비전 연결: positional encoding(사인/코사인), Fourier feature mapping.

### 4.16 noise and denoising intuition
- 수식 중심 요약: 관측 `y = x + n` 또는 `y = h*x + n`.
- 직관 중심 요약: 진짜 신호와 랜덤/구조적 방해를 분리하는 문제.
- 비전 연결: 촬영 노이즈(shot/read), 저조도 복원, diffusion/denoising 모델.

## 5. 이미지와 CNN으로 연결

### 5.1 Edge detection
- Sobel/Prewitt는 고주파 강조(high-pass) 필터로 이해 가능.
- CNN 초기 레이어가 edge-like 커널을 학습하는 이유와 연결된다.

### 5.2 Smoothing / Deblurring
- 블러는 보통 low-pass 시스템.
- deblurring은 역필터 또는 정규화 기반 복원 문제(노이즈 증폭과 trade-off).

### 5.3 Frequency bias in CNN
- CNN은 저주파 신호를 상대적으로 더 쉽게 학습하는 경향이 보고됨.
- 데이터 증강/손실 설계에서 고주파 성분 보존 전략이 필요할 수 있다.

### 5.4 Anti-aliasing in modern vision models
- stride/pooling 이전 저역통과 필터를 넣으면 aliasing 완화.
- 다운샘플링 안정성과 shift consistency 개선에 도움.

### 5.5 Fourier-based image analysis
- FFT로 artifact/노이즈/주기성 문제를 빠르게 진단 가능.
- 주파수 도메인 특징과 공간 도메인 특징을 함께 쓰는 하이브리드 모델 설계 가능.

## 6. 자주 틀리는 포인트
1. convolution과 correlation을 혼용해 구현 오차를 내는 경우
2. FFT 기반 처리에서 zero-padding/경계조건을 무시해 ringing 발생
3. downsampling 전에 anti-aliasing 필터를 생략
4. low-pass를 "무조건 좋은 denoising"으로 오해(디테일도 손실)
5. PSNR만 보고 perceptual quality를 놓침
6. 샘플링 정리를 시간축에만 적용하고 공간축(이미지)에는 잊는 경우
7. 주파수 도메인 해석에서 phase 정보를 과소평가

## 7. 구현/실험 아이디어
```python
import torch
import torch.nn.functional as F

# 1) Anti-aliasing 비교 실험 (toy)
x = torch.randn(1, 1, 128, 128)
down_raw = x[:, :, ::2, ::2]

# 간단한 3x3 low-pass 후 다운샘플
k = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
k = (k / k.sum()).view(1,1,3,3)
x_lp = F.conv2d(x, k, padding=1)
down_lp = x_lp[:, :, ::2, ::2]

print("raw std:", float(down_raw.std()), "lp std:", float(down_lp.std()))
```

```python
import torch

# 2) Fourier magnitude 확인
img = torch.randn(1, 1, 128, 128)
Fimg = torch.fft.fft2(img)
mag = torch.log1p(torch.abs(Fimg))
print("freq magnitude mean:", float(mag.mean()))
```

```python
import torch

# 3) blur + noise + naive deblur toy
img = torch.randn(1,1,64,64)
k = torch.ones(1,1,5,5) / 25.0
blur = torch.nn.functional.conv2d(img, k, padding=2)
noisy = blur + 0.05 * torch.randn_like(blur)
print("noisy var:", float(noisy.var()))
```

## 8. 논문 읽기 연결 포인트
- "frequency", "spectral", "anti-aliasing", "deconvolution", "transfer function" 키워드를 보면 신호 관점으로 해석.
- 모델 성능 향상 claim이 주파수 대역 제어(고주파 보존, aliasing 억제)와 연결되는지 점검.
- 샘플링/리사이즈 단계의 구현 세부가 재현성에 큰 영향을 주는지 확인.
- denoising/deblurring 논문에서 forward model(`y=h*x+n`) 가정과 실제 데이터 간 차이 검토.
- Fourier 도메인 손실을 쓰는 경우 magnitude/phase 중 무엇을 학습하는지 구분.

## 9. GitHub 세부 문서 분할 제안
권장 구조:

```text
signals-and-systems/
  README.md
  docs/
    signals-and-systems-for-cv.md
  signals-and-systems/
    signals-basics.md
    convolution-lti.md
    fourier.md
    sampling.md
    vision-connections.md
```

파일 역할:
- `signals-basics.md`: 개념 지도, 이산/연속, LTI, impulse, poles/zeros
- `convolution-lti.md`: convolution, impulse response, filtering 설계, deblurring 연결
- `fourier.md`: Fourier series/transform/DFT, frequency response, modulation
- `sampling.md`: sampling theorem, aliasing, anti-aliasing, resize 실전 팁
- `vision-connections.md`: CNN/ViT/복원/검색/세그멘테이션 사례 연결

## 10. 핵심 질문 5개
1. 내 모델의 실패가 표현력 부족인지, 샘플링/필터링 문제인지 어떻게 구분할 것인가?
2. 다운샘플링 단계에서 aliasing을 억제하는 장치를 충분히 넣었는가?
3. 현재 손실/아키텍처가 고주파 디테일을 과도하게 버리고 있지 않은가?
4. 논문의 주파수 해석이 magnitude 중심인지 phase까지 다루는지 확인했는가?
5. 데이터 전처리(리사이즈, 보간, 노이즈 모델)가 실제 센서 물리와 정합적인가?
