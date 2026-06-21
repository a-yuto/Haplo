# Haplo 言語仕様 — 型システム・サンプル（§3.6〜§3.8）

## §3.6 型の文法

基本型：`f32`, `f64`, `Int`, `Bool`。テンソルは `Tensor[次元,...]`（既定 dtype は f32）。

**次元変数**：型注釈中の小文字 id。同じ変数の再利用で「同じ長さ」を強制。

```
matmul : Tensor[m,k] -> Tensor[k,n] -> Tensor[m,n]
```

**shape 算術**：次元を変数式で書ける（依存型の入口）。

```
concat  : Tensor[m] -> Tensor[n] -> Tensor[m+n]
flatten : Tensor[m,n] -> Tensor[m*n]
```

**値依存の型**（dependent function type）：実行時の値が型に現れる。

```
range : (n : Int) -> Tensor[n]      -- 値 n が戻り型の長さに
zeros : (s : Shape) -> Tensor[s]
```

通常の `A -> B` は引数が後ろの型に出ない非依存版の特例。

**パラメータ付き型別名**：

```
type Vec n = Tensor[n]
type Mat m n = Tensor[m,n]
```

**shape 推論**：多くの次元変数は省略可。トップレベル定義には型注釈を付けるのが慣習。

---

## §3.7 プログラム構造とレイアウト

- ファイル＝モジュール＝トップレベル定義の列。**定義順は自由**（前方参照・相互再帰可）
- **レイアウト：改行が区切り、インデントは飾り**（列の深さは意味を持たない）
  - `(` が開いている／行末が演算子のときは行継続
- エントリポイント：`main` を評価して結果を表示
- 副作用（IO）・REPL・`import` は将来拡張

> P0 実装注：現在は定義順に env を構築するため前方参照不可。P1 で two-pass 対応予定。

---

## §3.8 総合サンプル（線形回帰・Elm スタイル）

北極星プログラム。G3 達成の証明になる。

```
-- 線形回帰の1ステップ（完全不変）
x : Tensor[4, 3]
x =
    [1.0, 2.0, 3.0;
     4.0, 5.0, 6.0;
     7.0, 8.0, 9.0;
     1.0, 0.0, 1.0]
y : Tensor[4]
y = [1.0, 2.0, 3.0, 0.5]
lr : f32
lr = 0.01

predict : Tensor[n, d] -> Tensor[d] -> f32 -> Tensor[n]
predict feats w b =
    feats @ w + b

mse : Tensor[n] -> Tensor[n] -> f32
mse pred target =
    mean ((pred - target) ^ 2)

loss : Tensor[3] -> f32
loss w =
    mse (predict x w 0.0) y

step : Tensor[3] -> Tensor[3]
step w =
    w - lr * grad loss w

main : Tensor[3]
main =
    iterate (zeros [3]) 1000 step
```

このプログラムが動く = lexer/parser/テンソル/autodiff/反復が全てつながった状態。
