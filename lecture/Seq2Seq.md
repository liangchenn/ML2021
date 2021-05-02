- 輸入 一串向量，輸出一串向量



## Applications

### Grammar Checking

把樹狀的文法結構，利用向量表達：

e.g.

`(S. V. (S. VP...))`





### Multi-label Classification

文章的 tags (可能有多個)

機器自己決定輸出幾個tag數目





## Encoder

- input a n-length vector, and output vector with same lengths

- Usually, encoder contains multiple `blocks`
  - 每一個 block 之中都有
    - self-attention layer
    - FC layers

- 而在 transformer 之中，加入了一個設計
  - input 會再與 self-attention 的輸出做結合
  
  - 此叫做 ==**residual connection**==
  
    
  
  <img src='fig1.png' width='70%' height='70%'>

- 這邊會做一個 Layer normalization
  - 方式是看同一個dimension之中，來做正規化

- fully connected 裡面也有 residual connection 最後在做一次norm 才會是一個 block 的輸出





## Decoder

產生辨識的結果

- load encoded results
- `<BOS>` : begin of sentence token, 用 one-hot vector 來表達，代表開始
- Size V vector (常見字的向量  e.g. common 3000 characters)
  - 先經過一個 softmax 把字的機率算出來，最高的作為output
  - 低一個 ouput 會被丟回輸入項，叫做 `autoregressive`
    - 因此如果中間有錯誤結果，可能就讓後面錯誤



### Comparison

比較一下 decoder and encoder

<img src='fig2.png' width='75%'>

- Masked
  - 通常 self attention 會把所有的input看完
  - 但 masked 指的是，只看之前的input
    - e.g. When calculating `a2` , only considering `a1`, `a2`
    - 對 decoder 來說，`a3` 之後還沒有被生出來（在看a2時）

### Output Length

> How decoder decides output length ?

- 這邊會有另一個 特殊 token 叫做 `<EOS>` (end of sentance)





### Non-Autoregressive Model (NAT)

- 丟入一堆begin encoded result ，一次產生所有句子

> But how to decide the length of output when using NAT ?
>
> > A : Use another ==predictor== for output length
> >
> > A : Ignore tokens after END

- 好處是速度快(因為平行化)，且可以控制輸出的長度
  - 可以 manipulate output length
- 但 NAT 的表現較 AT 還要差 --> multi-modality



### How decoder read encoder's inputs ?

- cross attention

<img src='fig3.png' width='50%'>

### Loss

- **Cross entropy** between the distribution and the one-hot vector





## Training Tips

### Copy Mechanism 



### Guided Attention

- 要求機器在做attention時，是有固定方式的
- e.g. 語音合成時，是由左到右的。  <- monotonic attention ...



### Beam Search*(可以再查一下)

- 每一次都樹狀找最高分數的path
  - greedy decoding
- 但有可能一開始較差，但後面較好
  - 但無法合理地找尋所有的path
  - beam search  就是用來解決此問題
- 有時候有用？有時候不行？
  - 通常在有固定正確答案的學習問題下，bs 表現較好



### BLEU score

- loss function 雖然 使用 cross entropy 但在 model selection 是以 BLEU score 來選擇
  - BLEU 無法作微分，因此不拿來作為損失函數
  - 可能可以放進 Reinforcement Learning 來做



### Exposure Bias

- 因為機器在 訓練時，是看到 ground truth 資料，而不是之前輸出的。
- 但在測試時，則沒有真實資料
- 這樣可能產生一步錯，步步錯的狀況
- 可以加一些noise來訓練解決