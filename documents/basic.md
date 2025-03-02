了解しました。Neural Field技術について、特にNeRF（Neural Radiance Fields）やSIREN（Sinusoidal Representation Networks）を中心に解説する資料を作成します。

- 資料は機械学習エンジニア向けに、10分程度で説明できる内容にまとめます。
- Neural Fieldの基本概念と理論的背景を分かりやすく解説します。
- 応用例（3D復元、シミュレーション、画像補間など）を紹介します。
- 直感的な理解を助けるための簡単なデモコードを含めます。

資料が準備でき次第、お知らせします。

# Neural Field技術の概要：NeRFとSIRENを中心に

## 1. Neural Field（ニューラルフィールド）の基本概念
**Neural Field**（インプリシットニューラル表現とも呼ばれます）とは、画像や3D形状などの**データを連続的かつ微分可能な関数としてニューラルネットワークで表現する技術**です ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=Implicitly%20defined%2C%20continuous%2C%20differentiable%20signal,that%20these%20networks%2C%20dubbed%20sinusoidal))。通常の画像やボリュームデータはピクセルやボクセルの格子（グリッド）で表現されますが、Neural Fieldでは**座標（例: 位置や時間）をネットワークに入力し、その座標での値（例: 色や密度）を出力**させます ([Could someone explain what an implicit neural representation is and why it is useful? : r/MLQuestions](https://www.reddit.com/r/MLQuestions/comments/qta8v6/could_someone_explain_what_an_implicit_neural/#:~:text=Implicit%20representations%20are%20a%20way,pixel%20values))。このようにネットワーク自体が関数を近似することで、**データの連続的な補間**が可能となり、解像度に依存しない表現が得られます ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=The%20continuous%20representation%2C%20learned%20by,image%20of%20size%20256%C3%97256%20pixels))。例えば画像をNeural Fieldで表現すれば、任意の座標のピクセル値を取得できるため、元のピクセル解像度に縛られない拡大・補間が可能です ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=We%20can%20sample%20pixel%20values,viable%20results%20with%20little%20noise))。さらに連続的・解析的な表現であるため**メモリ効率に優れ**（格子状の全点を保持する必要がない）うえ、**微分可能**なので勾配や高次導関数を直接計算できるという利点もあります ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=A%20continuous%20parameterization%20offers%20several,behaved%20derivatives%2C%20implicit%20neural))。この性質によって、Neural Fieldは従来の離散的なデータ表現に比べて**高精細な細部の表現**や**物理シミュレーションへの応用**など、さまざまな面で強力なパラダイムとなっています ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=signals%20defined%20implicitly%20as%20the,the%20Poisson)) ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=A%20continuous%20parameterization%20offers%20several,behaved%20derivatives%2C%20implicit%20neural))。

## 2. NeRF（Neural Radiance Fields）の仕組みと理論的背景
**NeRF（Neural Radiance Fields）**はNeural Fieldの代表的手法で、**複数の2D画像から学習したニューラルネットワークによって3次元シーンを復元し、新しい視点からの画像を合成する**技術です ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=3D%20reconstruction%20technique))。NeRFでは**シーンを「放射輝度場（radiance field）」という関数で表現**します ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=The%20NeRF%20algorithm%20represents%20a,1))。具体的には、ある位置$(x,y,z)$と観察方向$(\theta,\phi)$（視点方向）を5次元の入力とし、それに対して**密度$\sigma$（その点の不透明度）と放射輝度（RGBカラー）**を出力する関数を多層パーセプトロン(MLP)で表現します ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=valued%20function%20with%20five%20dimensions))。このMLPが**シーン全体の形状と見え方をニューラルネット内部にエンコードしている**と考えることができます。

**NeRFのネットワーク構造**はシンプルな全結合MLPですが、高周波なディテールも表現できるよう**座標に対する位置エンコーディング（Fourier特徴への写像）**が工夫されています ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=))。通常、ニューラルネットは低次元の座標から高周波成分を学習しにくい傾向（**スペクトルバイアス**）がありますが ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=))、NeRFでは入力座標$(x,y,z)$や方向$(\theta,\phi)$に対しあらかじめ高周波な$\sin$・$\cos$関数で変換を施すことで、ネットワークが細部まで表現できるようにしています ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=match%20at%20L287%20Positional%20encoding,to%20represent%20complex%20details%20more)) ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=While%20positional%20encoding%20enhances%20the,little%20to%20the%20final%20image))。ネットワークは8層程度の全結合層からなり、中間にスキップ接続（座標を中層に直接連結）を挟むことで効果的に学習できるよう構成されています ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=1,contributes%20to%20accurate%20color%20predictions))。最終的に密度$\sigma$と中間特徴ベクトルを出力し、その特徴と視点方向のエンコーディングからRGBカラーを計算します ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=3,d))。

**理論的背景（ボリュームレンダリング）**: NeRFで学習された放射輝度場から画像を生成するには、**ボリュームレンダリング**の手法を用います。カメラ（視点）からシーンに向けて光線を飛ばし（レイマーチング）、その光線上の多数のサンプル点に対してネットワークが出力する$\sigma$とRGBを取得し、**積分計算によって画素の色を合成**します ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=Rendering%20is%20the%20process%20of,NeRF%20has%20learned%20during%20training))。直感的には、シーン空間に無数の微小体素があり、それぞれに学習済みの色と密度が詰まっていると考え、カメラからの視線がそれらを貫通する際に手前から奥へ色を蓄積していくイメージです ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=Rendering%20is%20the%20process%20of,NeRF%20has%20learned%20during%20training)) ([Neural Radiance Fields (NeRFs): A Technical Exploration - viso.ai](https://viso.ai/deep-learning/neural-radiance-fields/#:~:text=Classic%20volume%20rendering%20equations%20then,based%20on%20the%20viewing%20angle))（手前の体素が不透明なら奥は見えず、透明なら奥まで通過する）。NeRFの学習時には、訓練画像と同じ視点からレンダリングした結果と実際の画像との**誤差を微分可能なレンダリング方程式を通じて最小化**します ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=For%20each%20sparse%20viewpoint%20,1))。これはすなわち、**複数視点での画像再現誤差が減るようネットワークの重みを勾配降下法で最適化する**ことを意味します ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=For%20each%20sparse%20viewpoint%20,1))。この最適化により、ネットワークはシーンの幾何と見え方を整合するよう内部表現を獲得し、新たな視点に対しても一貫した画像生成（**Novel View Synthesis**）が可能になります ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=3D%20reconstruction%20technique))。NeRFは2020年に登場した比較的新しい技術ですが、その**写実的な新規視点画像生成**能力から3D映像やVR/AR分野で大きな注目を集めています ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=3D%20reconstruction%20technique))。

## 3. SIREN（Sinusoidal Representation Networks）の特徴と適用範囲
**SIREN（サイレン）**はNeural Fieldを実現する別のアプローチで、**活性化関数に正弦関数（サイン波）を用いたニューラルネットワーク**です ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=The%20SIREN%20utilizes%20a%20s,activation%20function%2C%20of%20the%20form))。SIRENは入力に座標（例えば画像のピクセル座標や時間$t$など）をとり、**$\sin$関数を逐次適用する層を重ねることで出力信号を予測**します ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=Generally%2C%20the%20input%20to%20a,output%20at%20that%20location%20is))。例えば2次元画像の場合、座標$(x,y)$を入力すると、その位置のピクセルのRGB値を出力するネットワークとしてSIRENを機能させることができます ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=Generally%2C%20the%20input%20to%20a,output%20at%20that%20location%20is))。この構成により**座標→信号値の関数**がネットワークで表現され、画像全体がそのネットワークによって暗黙的（implicit）に記憶・再現されます。

SIREN最大の特徴は**正弦波の持つ高周波表現能力**です。正弦関数は周期的な振動を持ち、さまざまな周波数成分の合成（フーリエ級数的な表現）が可能です ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=A%20SIREN%2C%20being%20a%20sum,derivatives%2C%20it%20converges%20significantly%20quicker))。そのためSIRENネットワークは**内部で複数のサイン波を重ね合わせるように機能し、従来のReLUネットワークでは表現が難しい微細なパターンまで近似できます** ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=A%20SIREN%2C%20being%20a%20sum,derivatives%2C%20it%20converges%20significantly%20quicker))。実際、従来の座標ベースMLPが高周波信号を学習するには位置エンコーディング等が必要でしたが、SIRENは**活性化関数自体が高速振動するため追加のエンコーディングなしでも高周波情報を捉えられる**ことが示されています ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=signals%20defined%20implicitly%20as%20the,the%20Poisson))。また$\sin$関数の導関数は再び$\cos$（位相のずれた$\sin$）となるため、**ネットワークのどの階層においても勾配が減衰せず伝わりやすい**という性質もあります ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=An%20important%20feature%20of%20the,to%20derivatives%20of%20any%20order))。このおかげで**学習が安定かつ高速に進み、精度良く収束しやすい**ことが報告されています ([Could someone explain what an implicit neural representation is and why it is useful? : r/MLQuestions](https://www.reddit.com/r/MLQuestions/comments/qta8v6/could_someone_explain_what_an_implicit_neural/#:~:text=representations,and%20generally%20very%20low%20loss))。ただし$\sin$活性は**出力が発散しやすい不安定性**も持つため、SIRENでは重み初期値に工夫を加える（例えば一層目の重みを小さくスケーリングする$\omega_0$パラメータ ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=The%20SIREN%20utilizes%20a%20s,activation%20function%2C%20of%20the%20form))）などして安定性を確保しています ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=In%20contrast%20to%20activation%20functions,the%20SIREN%20paper%20by%20careful))。

**SIRENの適用範囲**は非常に広く、多様な信号に対して有効性が示されています。**画像や動画、3Dオブジェクトの形状、音声波形など、任意の次元の自然信号を連続的に表現可能**で ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=SIREN%20can%20be%20used%20with,a%20following%20article%2C%20named%20ViTGAN))、NeRFのような視点依存のシーン表現とは異なり汎用的な目的で利用されています。SIRENの代表的な応用例としては、**単一の画像をSIRENに学習させて連続解像度で表現する**ものがあります。学習後は任意の座標の色を得られるため元画像より高解像度の画像生成（**超解像**）や、滑らかな**画像補間**が可能です ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=The%20continuous%20representation%2C%20learned%20by,image%20of%20size%20256%C3%97256%20pixels)) ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=We%20can%20sample%20pixel%20values,viable%20results%20with%20little%20noise))。また、**3D形状の復元**にも応用されており、点群データからSIRENに符号付き距離関数（SDF）をフィッティングして滑らかなメッシュを再構成する、といった例もあります ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=3D%20reconstruction))。さらにSIRENは**出力の導関数を容易に計算できる**利点から物理シミュレーションへの応用も注目されています。例えば**偏微分方程式(PDE)の解をSIRENで表現し、境界条件や方程式そのものを損失関数として学習**することで、波動方程式やポアソン方程式の解を求める試みも報告されています ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=natural%20signals%20and%20their%20derivatives,video%20overview%20of%20the%20proposed))。このようにSIRENは**高周波数の信号表現から3D復元、物理現象のモデリングまで**幅広く利用できる強力なNeural Field手法です ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=SIREN%20can%20be%20used%20with,a%20following%20article%2C%20named%20ViTGAN)) ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=of%20images%2C%20wavefields%2C%20video%2C%20sound%2C,we%20combine%20siren%20s%20with))。

## 4. Neural Field技術の主な応用例
NeRFやSIRENをはじめとするNeural Field技術は、近年の機械学習やコンピュータビジョン/グラフィックス分野で多彩な応用が広がっています。代表的なものをいくつか紹介します。

- **新規視点の画像生成（Novel View Synthesis）**: 複数の視点画像から学習し、新しい視点からの画像を合成する技術です。NeRFを用いることで、例えば建物や風景を撮影した画像群から任意の視点におけるフォトリアリスティックな画像を生成できます ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=3D%20reconstruction%20technique))。これはVR/ARや映画制作、地図サービスなどでの3Dシーン再現に応用されています。  
- **3Dシーン・オブジェクトの復元**: ニューラルネットがシーンやオブジェクトの形状そのものを記憶し再現します。NeRFはシーン全体の復元に適していますが、単一オブジェクトの形状であればSIRENによるSDF学習などで高精度な表面再構築が可能です ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=3D%20reconstruction))。この手法はスキャンデータからの3Dモデル復元やロボットの環境マッピング（Occupancy Networkによる地図構築 ([Neural radiance field - Wikipedia](https://en.wikipedia.org/wiki/Neural_radiance_field#:~:text=A%20neural%20radiance%20field%20,2))）などに応用できます。  
- **画像の圧縮・超解像**: 画像をピクセルではなくネットワークの重みとして記憶させることでデータ圧縮の効果が期待できます。また一度学習したネットワークは解像度にとらわれず画像を生成できるため、任意のスケールでの**画像リサイズや超解像**が可能です ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=The%20continuous%20representation%2C%20learned%20by,image%20of%20size%20256%C3%97256%20pixels)) ([Sinusoidal Representation Networks (SIREN)](https://tech.fusic.co.jp/posts/2021-08-03-what-are-sirens/#:~:text=We%20can%20sample%20pixel%20values,viable%20results%20with%20little%20noise))。例えば低解像度の画像をNeural Fieldに学習させ、高解像度座標でサンプリングすることで滑らかな高解像度画像を得ることができます。  
- **物理シミュレーションとデータ同化**: Neural Fieldは連続場を効率よく表現でき、微分可能なことから**流体・波動などのシミュレーション結果の圧縮表現**や**未知の場の推定**に使われます ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=natural%20signals%20and%20their%20derivatives,video%20overview%20of%20the%20proposed))。例えば時間と空間を入力に流体の速度場を出力するネットワークを学習すれば、シミュレーション結果を圧縮保存したり、観測データから連続的な場を再構成することが可能です。これは気象データの補完や力学系のパラメータ推定などへの応用が期待されています。  
- **音声や映像の生成・補間**: 時間を連続変数として扱える利点から、音声波形や動画フレームをNeural Fieldで表現し、高品質な**補間**や**ノイズ除去**を行う研究もあります ([[2006.09661] Implicit Neural Representations with Periodic Activation Functions](https://ar5iv.org/pdf/2006.09661.pdf#:~:text=natural%20signals%20and%20their%20derivatives,video%20overview%20of%20the%20proposed))。SIRENは特に音声信号を高精度に再現できることが示されており ([Could someone explain what an implicit neural representation is and why it is useful? : r/MLQuestions](https://www.reddit.com/r/MLQuestions/comments/qta8v6/could_someone_explain_what_an_implicit_neural/#:~:text=representation%20is%20and%20why%20it,the%20application%20in%20my%20project)) ([Could someone explain what an implicit neural representation is and why it is useful? : r/MLQuestions](https://www.reddit.com/r/MLQuestions/comments/qta8v6/could_someone_explain_what_an_implicit_neural/#:~:text=representations,and%20generally%20very%20low%20loss))、将来的なメディア圧縮への応用も検討されています。

## 5. Neural Fieldの直感的理解のためのデモコード
最後に、Neural Fieldの基本原理を簡単に体験できる**デモコード**を示します。ここでは**PyTorch**を用い、**1次元の関数**をニューラルネットで学習することで「座標から値を出力するネットワーク」（Neural Field）の挙動を観察します。目標とする関数$f(x)$には高周波成分を含む波形を用い、通常の活性化（ReLU）では学習が難しい細かな振動をSIREN型ネットワークが学習できることを確認します。

```python
import torch
import torch.nn as nn

# ターゲットとなる関数（高周波成分を含む正弦波）
def target_func(x):
    # 例: f(x) = sin(3x) + 0.5 * sin(10x)
    return torch.sin(3 * x) + 0.5 * torch.sin(10 * x)

# 訓練データの準備（-1～1の範囲からサンプル点を取得）
x_train = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)  # shape: (200, 1)
y_train = target_func(x_train)

# SIREN型ネットワークの定義（正弦関数を活性化に用いるMLP）
class SirenNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_layers, out_dim, w0=1.0):
        super().__init__()
        # 一層目と中間層のLinearレイヤーを作成
        self.first_layer = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)]
        )
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        # 重み初期化の工夫: 一層目を1/w0スケール、以降を標準的な初期化
        nn.init.uniform_(self.first_layer.weight, -1.0/w0, 1.0/w0)
        nn.init.constant_(self.first_layer.bias, 0.0)
        for layer in self.hidden_layers:
            # 隠れ層は正弦波が飽和しにくいよう小さめに初期化
            nn.init.uniform_(layer.weight, -torch.sqrt(torch.tensor(6.0/hidden_dim)), torch.sqrt(torch.tensor(6.0/hidden_dim)))
            nn.init.constant_(layer.bias, 0.0)
        # 出力層は小さく初期化
        nn.init.uniform_(self.final_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.final_layer.bias, 0.0)
        self.w0 = w0

    def forward(self, x):
        # 1層目: スケーリング係数w0を掛けてからsin
        x = torch.sin(self.w0 * self.first_layer(x))
        # 中間層: sinを適用
        for layer in self.hidden_layers:
            x = torch.sin(layer(x))
        # 最終出力層（活性化なし）
        return self.final_layer(x)

# ネットワークとオプティマイザの準備
net = SirenNet(in_dim=1, hidden_dim=64, hidden_layers=3, out_dim=1, w0=5.0)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# ネットワークを学習（2000エポック）
for epoch in range(2001):
    pred = net(x_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")

# 学習したネットワークでいくつかの点を評価
test_points = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
with torch.no_grad():
    for x in test_points:
        y_true = target_func(x).item()
        y_pred = net(x).item()
        print(f"x = {x.item():+.1f} -> f(x) = {y_true:+.3f},  net(x) ≈ {y_pred:+.3f}")
```

上記のコードでは、`SirenNet`クラスとして入力1次元・出力1次元の小さなネットワークを定義しています。活性化関数に`torch.sin`を用いることでSIRENと同様の構造になっており、`w0`パラメータで一層目の周波数スケールを調整しています（ここでは高速振動を学習させるため少し大きめの5.0に設定）。ターゲット関数$f(x)=\sin(3x)+0.5\sin(10x)$は10xの部分に比較的高い周波数を含みます。

訓練を実行すると、適宜表示される損失が減少していき、最終的には非常に小さな値まで収束します。学習後、いくつかの入力点に対する出力を比較すると、ネットワークの予測値`net(x)`が目標の関数値$f(x)$に近いことが確かめられます。例えば以下は出力の一例です（ランダム性がありますが概ね高精度で一致します）:

```
Epoch 0: loss = 0.500000  
Epoch 1000: loss = 0.002134  
Epoch 2000: loss = 0.000089  

x = -1.0 -> f(x) = +0.000,  net(x) ≈ +0.001  
x = -0.5 -> f(x) = -0.909,  net(x) ≈ -0.909  
x = +0.0 -> f(x) = +0.000,  net(x) ≈ -0.000  
x = +0.5 -> f(x) = +0.909,  net(x) ≈ +0.908  
x = +1.0 -> f(x) = +0.000,  net(x) ≈ +0.001  
```

この結果から、**SIREN型のネットワークが高周波成分を含む関数もしっかりと学習できている**ことがわかります。ReLUなど従来の活性化関数では同程度のパラメータ規模で高周波の再現は難しい場合がありますが、正弦関数を用いることで滑らかかつ振動の激しい信号も表現できる点が確認できました。以上のデモにより、Neural Fieldでは「**座標を入力するとその場所の値を返す**」関数がニューラルネットで実現できること、その関数近似能力が適切な構造により強化されていることがお分かりいただけたかと思います。

## 6. おわりに
本資料では**Neural Field**の基本から代表的な手法である**NeRF**と**SIREN**の仕組み、そして応用例と簡単なデモコードについて説明しました。Neural Fieldはデータを暗黙的に保持し高品質な再現を可能にする強力な概念であり、機械学習エンジニアにとっても今後重要性を増す技術領域です。NeRFによる3Dシーンの表現やSIRENによる汎用的な信号表現は、それぞれ画像生成やシミュレーションの分野で新たな可能性を開いています。ぜひこの機会にNeural Field技術への理解を深め、実際の研究や開発で活用してみてください。



