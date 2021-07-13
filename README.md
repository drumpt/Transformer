# Transformer
This repository is for implementing Transformer neural machine translation model.

## :notebook_with_decorative_cover: Dataset
- [Multi30k](https://www.aclweb.org/anthology/W16-3210/)

## :pencil2: Requirements
```bash
CUDA==10.1
cuDNN==7.6.5
Python==3.8.6
torch
matplotlib
tqdm
kora
```

## :computer: Usage
### Train and validation
Run the cells in main.ipynb in order

## :question: Implementation Details
- Training and validation 구현
- train시 \<eos\> token 뒤에 오는 token은 학습에 포함되면 안됨. batch로 만들려면 모두 같은 차원을 가져야하기 때문에 문장을 잘라서 뒤에 padding을 붙여 같은 길이로 만듦
- test시 \<sos>와 \<pad\>는 제외하고 생성하게 만듦
- test시 \<eos\>가 모든 문장에 나타난 경우나 정해준 max length에 도달했을 때 종료
- Positional encoding 방법 개선
- GPU 이용 가능하게 만들기
  - model과 src_batch, tgt_batch에 모두 .to(device)를 추가
  - model이 submodel로 구성된 경우 forward에서는 .to(device)를 추가할 필요 없고, 정의된 모든 model과 submodel의 __init__에서 각각의 모델에 .to(device)를 추가하며 됨
  - 모델 내무에서 torch 연산 함수가 아닌 다른 함수를 이용하는 경우 torch의 변수와 상호작용하는 변수에 모두 .to(device)를 추가해줌
  - 다른 함수에서 tensor가 호출되는 경우 epoch가 끝나도 없어지지 않을 수 있으니 다른 함수의 파라미터로 tensor를 주기 전에 cpu로 이전시킴
- conda error 해결

## :goal_net: TODOs
- final layer 분석
- positional encoding 재확인

## :moyai: References
- Attention Is All You Need (https://arxiv.org/abs/1706.03762)
- Transformer in DGL (https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer)
