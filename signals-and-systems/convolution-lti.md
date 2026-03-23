# Convolution and LTI for Images and CNNs

## 1. 목적
convolution, impulse response, filtering을  
이미지 복원/인식/CNN 설계에 바로 연결하는 실전 레퍼런스.

## 2. 수식 중심 요약
- 1D: `y[n] = sum_k x[k] h[n-k]`
- 2D: `y[i,j] = sum_m sum_n x[m,n] h[i-m, j-n]`
- 주파수 영역: `Y(omega) = X(omega) H(omega)`
- LTI 시스템에서 컨볼루션은 완전한 입력-출력 기술

## 3. 직관 중심 요약
- 커널 `h`는 시스템의 작동 규칙
- 컨볼루션은 "주변 정보를 가중합"하는 로컬 규칙
- 주파수 영역에서는 성분별 gain 조정으로 해석 가능

## 4. 비전 연결

### 4.1 edge detection
- 고주파 강조(high-pass) 필터
- Sobel/Prewitt/Laplacian은 공간 미분 근사

### 4.2 smoothing
- low-pass 필터로 노이즈 억제
- Gaussian blur는 고주파 잡음을 줄이지만 경계 디테일도 약화

### 4.3 deblurring
- 블러 모델: `y = h*x + n`
- 역문제 성격: 고주파 복원 시 노이즈가 같이 증폭되기 쉬움
- regularization(Tikhonov, TV 등)이 필요한 이유

### 4.4 CNN 해석
- conv layer는 학습 가능한 필터뱅크
- 초기 레이어는 edge/texture 성분, 후반은 의미 특징 추출 경향
- stride conv는 downsampling과 filtering을 동시에 수행

## 5. filter design 관점
- low-pass: smoothing/anti-aliasing
- high-pass: edge/detail 강화
- band-pass: 특정 texture 주파수 강조
- notch: 주기적 노이즈 제거

실무 팁:
- 필터 목적(노이즈 제거 vs 디테일 보존)을 먼저 고정
- 필터 크기/경계 처리에 따른 artifact를 반드시 시각화

## 6. 논문 읽기 포인트
1. convolution이 standard인지 depthwise/group/dynamic인지
2. blur 모델의 kernel이 고정인지 학습인지
3. 경계 조건(padding) 가정이 명시됐는지
4. inverse filtering 계열에서 노이즈 안정화 전략이 있는지
5. 공간 도메인과 주파수 도메인 손실이 어떻게 결합되는지

## 7. 구현 체크리스트
1. convolution vs correlation convention 확인
2. padding 모드(`zeros`, `reflect`, `replicate`) 비교
3. stride 전 anti-aliasing 필터 적용 여부 점검
4. deblurring 실험 시 noise level 시나리오 분리
5. kernel 시각화 + frequency response 시각화 동시 기록

## 8. 미니 실험 아이디어
```python
import torch
import torch.nn.functional as F

img = torch.randn(1,1,128,128)
sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
edge = F.conv2d(img, sobel_x, padding=1)

gauss = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
gauss = (gauss/gauss.sum()).view(1,1,3,3)
smooth = F.conv2d(img, gauss, padding=1)

print("edge std:", float(edge.std()), "smooth std:", float(smooth.std()))
```

## 9. 다음 문서
- [fourier](./fourier.md)
- [sampling](./sampling.md)
- [vision-connections](./vision-connections.md)
