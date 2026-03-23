# Signals Basics for Vision Research

## 1. 목적
비전 연구자가 신호및시스템을 "교과서 공식"이 아니라  
이미지 모델링, 데이터 전처리, CNN 동작 해석에 바로 쓰기 위한 기준 문서.

## 2. 수식 중심 요약
- 시스템: `y = H{x}`
- 연속/이산: `x(t)` vs `x[n] = x(nT_s)`
- LTI: `H{a x1 + b x2} = aH{x1}+bH{x2}`, `H{x[n-n0]} = y[n-n0]`
- 임펄스 응답: `h[n] = H{delta[n]}`
- 컨볼루션: `y[n] = sum_k x[k] h[n-k]`
- 전달함수:
  - 연속: `H(s)` (Laplace)
  - 이산: `H(z)` (Z-transform)
- 극/영점: `H(z)=B(z)/A(z)`에서 zero는 `B(z)=0`, pole은 `A(z)=0`

## 3. 직관 중심 요약
- 시스템은 "입력 이미지를 어떤 규칙으로 바꾸는 연산자"다.
- LTI는 "섞어도 되고, 위치를 옮겨도 규칙이 같음"을 뜻한다.
- 임펄스 응답은 시스템의 지문이다.
- pole/zero는 "무엇을 강조하고 무엇을 억제하는가"를 요약한다.

## 4. 비전/CNN 연결

### 4.1 이미지 신호 관점
- 이미지: 2D 이산 신호 `x[m,n]`
- 비디오: 시간까지 포함한 3D 신호 `x[t,m,n]`
- 멀티모달(깊이/IR): 다채널 신호로 확장

### 4.2 시스템 관점 파이프라인
- 광학/센서/ISP/압축/네트워크를 연속된 시스템 체인으로 모델링 가능
- 성능 이슈를 단계별 시스템 문제로 분해 가능:
  - blur: optics/손떨림 시스템
  - noise: 센서 시스템
  - aliasing: sampling 시스템

### 4.3 CNN과 LTI의 관계
- CNN의 선형 부분(conv)은 LTI와 직접 연결
- 다만 ReLU/BN/attention 등 비선형이 중간에 들어가 전체는 비선형 시스템
- 연구 실무에서는 "선형 근사 구간"과 "비선형 구간"을 분리해 분석하면 디버깅이 빨라진다

## 5. Laplace/Z-transform/pole-zero를 왜 비전에서 보나
- Laplace: 연속 시간 모델(모션 블러/카메라 진동)의 이론 해석
- Z-transform: 디지털 필터/재귀 연산 안정성 분석
- pole-zero: 필터 응답의 안정성/리플/ringing 해석

비전 예시:
- deblurring에서 역필터가 불안정한 이유를 pole 구조로 설명 가능
- video denoising에서 temporal filter 안정 조건을 pole 반경으로 설명 가능

## 6. 논문 읽기 포인트
1. "forward model"이 `y = h*x + n` 형태로 가정되는지 확인
2. 연속 모델인지 이산 모델인지 구분
3. 시스템이 LTI 가정인지, space-variant인지 확인
4. stability 관련 논의가 pole/gradient explosion 관점과 연결되는지 확인
5. 구현 단계에서 sampling/resize가 이론 가정과 일치하는지 확인

## 7. 구현 체크리스트
1. 신호 축 정의(`H,W,C` 또는 `T,H,W,C`)를 문서화
2. convolution/correlation convention 통일
3. 경계 조건(padding, reflection, circular) 명시
4. 수치 안정성(정규화, epsilon, clipping) 점검
5. 실험 로그에 frequency-domain 통계를 함께 저장

## 8. 미니 실험 아이디어
```python
import torch
import torch.nn.functional as F

# impulse response 확인
k = torch.tensor([[0., -1., 0.],
                  [-1., 5., -1.],
                  [0., -1., 0.]]).view(1,1,3,3)

delta = torch.zeros(1,1,21,21)
delta[...,10,10] = 1.0
h = F.conv2d(delta, k, padding=1)
print("impulse response center:", float(h[...,10,10]))
print("sum(h):", float(h.sum()))
```

## 9. 다음 문서
- [convolution-lti](./convolution-lti.md)
- [fourier](./fourier.md)
- [sampling](./sampling.md)
- [vision-connections](./vision-connections.md)
