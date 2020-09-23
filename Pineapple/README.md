# Pineapple    
実行環境：    

Windows10    
pythonのバージョン: 3.7.9    
tensorflowのバージョン: 2.2.0    
GPU:  Goforce RTX 2070    
 
１．データ準備    
Root path:    
C:\Users\zenkori\Documents\Pineapple（PCのpath環境によってここのpath違います）    
（zenkoriのどころは個人のpathのusernameに変更する必要があります）    

1-1CROP    
- 実行ディレクトリ    
- root\data    

- 実行方法    
- python pineapplecrop.py    

- 入力ファイル    
Pineapple.jpg    

- 出力ファイル    
n.jpg（nのところは数字が入ります。）    

パイナップル全形の写真からトレーニング用のcrop写真を作ります    
１５行のrange()の引数で写真の枚数を変えることができます    
２０、２１行のところでcropサイズ変更できます    

1-2ＶＡＥデータ配置    

入力ファイ:　pineapple.jpg    

入力データ置き場：    	

作成したトレーニング用写真をroot\data\train にと　評価用写真をroot\data\valに置いて    　
学習結果テストの写真をroot\data\testに置きます    

出力データ置き場： 復元画像とheatmap出るどころはroot\data\test     

1-3 リサイズ    
学習結果テストの写真を、VAEの出力サイズに合わせるため、root\data\resize.pyで一旦リサイズする    

- 実行ディレクトリ    
root\data    

- 実行方法    
- python pineapplecrop.py        

- 入力ファイル    
テスト画像ファイル    
- 出力ファイル    
リサイズされたテスト画像ファイル(pineapple256.jpg)    

2.トレーニング    
- 実行ディレクトリ    
root\AnormalyDetection\VAE_1.0\    

- 実行方法    
１３行と１４行のtrain_dirとtest_dir　pathをPCの環境に合わします    
 python Train.py    

- 入力ファイル    
root\data\trainとroot\data\valの中に作成した画像ファール    

- 出力ファイル    
モデルのjson :  root\AnormalyDetection\VAE_1.0\ae_pinapple.json    
Weight file:  root\AnormalyDetection\VAE_1.0\ae_weights.h5    

3.インフャー    
- 実行ディレクトリ    
root\AnormalyDetection\VAE_1.0\    

- 実行方法
１６行と　data_pathをPCの環境に合わせます    
１８行のjpg_filesにテスト写真のファイル名をリストに追加します    
python　infer.py

- 入力ファイル    
テスト画像:　　　リサイズされたテスト画像ファイル(pineapple256.jpg)    
モデルのjson :  root\AnormalyDetection\VAE_1.0\ae_pinapple.json    
Weight file:  root\AnormalyDetection\VAE_1.0\ae_weights.h5    

- 出力ファイル    
結果画像　root\data\test である    
