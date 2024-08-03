import cv2
import numpy as np
import matplotlib.pyplot as plt

#対象画像
img = cv2.imread('./S__2646018.jpg')
#テンプレート画像
templ = cv2.imread('./S__2646020.jpg')

# グレースケール化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

#cv2.TM_CCOEFF_NORMED(正規化相互相関演算)によりパターンマッチング
result = cv2.matchTemplate(img_gray,templ_gray,cv2.TM_CCOEFF_NORMED)

print(result)
#類似度が閾値以上の箇所を抽出
threshold =0.38
match_y, match_x = np.where(result >= threshold)
print(match_x,match_y)

#テンプレート画像のサイズ
w = templ.shape[1]
h = templ.shape[0]

#元画像のコピー
dst = img.copy()

#マッチした箇所に赤枠を描画
for x,y in zip(match_x, match_y):
    cv2.rectangle(dst,(x,y),(x+w, y+h),(0,0,225),2)          

plt.imshow(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
plt.show()
