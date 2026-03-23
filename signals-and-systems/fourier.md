# Fourier for Vision and Deep Learning

## 1. 목적
Fourier series/transform/DFT를 비전 연구에서 실제로 쓰는 방식(분석/복원/강건성/가속)으로 정리.

## 2. 수식 중심 요약
- 연속 Fourier transform: `X(omega) = integral x(t)e^{-j omega t} dt`
- 역변환: `x(t) = (1/2pi) integral X(omega)e^{j omega t} d omega`
- DFT: `X[k] = sum_{n=0}^{N-1} x[n] exp(-j 2pi kn/N)`
- 컨볼루션 정리: 공간 컨볼루션 <-> 주파수 곱
- 변조(shift): 시간/공간 곱셈 <-> 주파수 이동

## 3. 직관 중심 요약
- 이미지는 주파수 성분의 합이다.
- 저주파: 밝기/대략 구조
- 고주파: 경계/세부 텍스처/노이즈
- magnitude는 "얼마나 강한 성분인지", phase는 "어디에 있는지"를 담는다.

## 4. 비전 연결

### 4.1 Fourier-based image analysis
- FFT로 주기적 노이즈/모아레/압축 artifact 진단
- 주파수 마스크로 특정 성분 억제/강조

### 4.2 frequency bias in CNN
- CNN이 상대적으로 저주파를 먼저 학습하는 경향 보고
- 고주파 보존이 중요한 task(복원, 초해상도)에서는 손실 설계가 중요

### 4.3 Fourier feature / positional encoding
- 좌표를 사인/코사인 기반 주파수 기저로 매핑해 고주파 표현력 향상
- NeRF류 표현 학습에서 핵심 기법

### 4.4 FFT-based acceleration
- 큰 kernel convolution에서 FFT 방식이 유리할 수 있음
- 실제 이득은 텐서 크기/하드웨어/메모리 패턴에 의존

## 5. modulation overview
- 변조는 정보를 다른 주파수 대역으로 이동
- 비전에서는 명시적 통신 변조보다 feature encoding 관점에서 자주 등장
- 예: sinusoidal embedding, complex-valued filtering

## 6. 논문 읽기 포인트
1. magnitude만 다루는지 phase도 다루는지
2. frequency loss가 어느 대역을 강조하는지
3. FFT 사용이 정확도 개선용인지 계산 가속용인지
4. 주파수 도메인 조작 후 공간 artifact(링잉) 평가가 있는지
5. train/eval에서 동일한 frequency pre-processing을 사용하는지

## 7. 구현 체크리스트
1. FFT 전 입력 정규화/zero-mean 여부 확인
2. log-magnitude 시 `log1p` 사용
3. frequency mask 적용 시 conjugate symmetry 유지 확인
4. 역변환 후 clipping/normalization 처리
5. phase 정보가 필요한 task인지 먼저 판단

## 8. 미니 실험 아이디어
```python
import torch

img = torch.randn(1,1,128,128)
Fimg = torch.fft.fft2(img)
mag = torch.log1p(torch.abs(Fimg))

# 간단한 low-pass mask
h, w = img.shape[-2:]
yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
cy, cx = h//2, w//2
r = ((yy-cy)**2 + (xx-cx)**2).float().sqrt()
mask = (r < 20).to(img.device).unsqueeze(0).unsqueeze(0)

Fshift = torch.fft.fftshift(Fimg)
F_lp = Fshift * mask
img_lp = torch.fft.ifft2(torch.fft.ifftshift(F_lp)).real

print("orig std:", float(img.std()), "lp std:", float(img_lp.std()))
```

## 9. 다음 문서
- [sampling](./sampling.md)
- [vision-connections](./vision-connections.md)
- [convolution-lti](./convolution-lti.md)
