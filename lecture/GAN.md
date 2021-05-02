# Generative Network

- 可以輸出 ==distribution== 的 network 
- 正常來說，輸入x輸出y

**流程**

給定一個 x 跟夠簡單的 distribution z (we know the structure f(x))，輸出一個 distribution

## Why ?

例如在做小精靈遊戲畫面預測，通常NN方式很可能會有同一個inputs，卻可以接受不同的結果

向左走跟向右走同時可以，因此單一輸出在此會不太好。

 

## Generative Adversarial Network

- unconditional generation
- conditional generation



### Unconditional

<img src='fig4.png' height='50%'>

- 只有z 的部分，而沒有x
- z ~ normal distribution
- generator : 產生很高維的向量



### Discriminator

- input a image
- output a ==scalar== denoting the score of fake 越高越好 [0, 1]
- 通常可以用 CNN 架構



## Algorithm

G : Generator ; D: Discriminator

- initialize params for G and D
- fix G, update D
  - 一個分類問題：區分出真正的人臉跟產生出來的

- fix D, update G
  - vector --> NN Generator --> D --> score


$$
G^* = \arg\min_G Div(P_G, P_{data})
$$

- 找一組 generator 讓 divergence of Pg 跟 真實資料越小越好

- 但 **divergence** 怎麼定義？？  *** 注意一下

  - negative cross entropy

    $V(G,D) = E_{y\sim data}[lnD(y)] + E_{y\sim G}[ln(1-D(y))]$

    

## Tips

1. **用不是 JS-divergence 來作為 loss**

- JS-divergence 可能的問題
  - data 之中 PG 跟 PData 兩個分布重疊很小
  - 或是 sampling 時，因為點不夠稠密，也會造成重疊很少
- 沒有重疊的話，JSD = ln(2) 
  - 而且如果沒有重疊， binary classifier 會硬背答案，
    達到 100% accuracy <- 這樣會不知道訓練過程是否越來越好

2. **Wasserstein distance**

- 把 P 移動到 Q 的平均距離（在只有一個點下，可以這樣理解）
  - 但把 P distribution 變成 Q distribution 時，有多種 moving plans
    - 窮舉 moving plans，來看最小的距離

$$
\max_{D \in 1-Lipschitz} \{ E_{x \sim P_{data}}[D(y)] - E_{x \sim P_{G}}[D(y)]\}
$$

- 1-Lipschitz : 足夠平滑的函數，不可變動過劇烈
  - 如果沒有重疊的話，不夠平滑的D會導致讓 generated, real 的值相差很大

<img src='fig5.png' width='80%'>

- spectral normalization could give us 1-Lipschitz function

## Challenges

- If either G or D fail to improve, then the network will fail



## Lecture 0416

有一種評估方式是，一樣去跑一個 image 辨識的模型，並產生一個 distribution

當機率分布$p(c|y)$平坦，則比較不好（not identifiable）

## Mode Collapse

- 會產生很多同樣pattern的圖片



# Conditional Generator

- Given $x$ and provide $z$ to generator, then output $y$

- e.g.  `text-to-image​`, `pix2pix/image translation`
  - input dog, output dog image
  - 需要 (text, image) paired data



# Unpaired Data GAN

- Unsupervised Learning
  - 不成對 x, y  <== unlabelled data
  - Solutions
    - pseudo labelling
    - back translation
  - 但上面還是需要一些成對的資料
- 但有時候會有完全沒有成對資料的狀況
  - e.g. 影像風格轉換



- Idea : input image in x domain, and output image in y domain

### Cycle GAN

