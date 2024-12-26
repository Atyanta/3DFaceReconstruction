# Rekonstruksi wajah 3D dengan full kulit: improvement dari DECA: Detailed Expression Capture and Animation (SIGGRAPH2021) oleh Atyanta

<p align="center"> 
<img src="TestSamples/teaser/results/teaser.gif">
</p>
<p align="center">input image, aligned reconstruction, animation with various poses & expressions<p align="center">

Bisa buka di COLAB!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OZC26ygYIXIPubZQf6E0eKz8rRmykjC6?usp=sharing)

Ini adalah cara memakainya! Namun tolong cek di google COLAB untuk mempermudah anda melakukan konfigurasinya! 

Cek definisi asli:
DECA reconstructs a 3D head model with detailed facial geometry from a single input image. The resulting 3D head model can be easily animated. Please refer to the [arXiv paper](https://arxiv.org/abs/2012.04012) for more details.

Apa yang bisa dilakukan:
* **Rekonstruksi:** menghasilkan pose kepala, bentuk, geometri wajah yang detail, dan informasi pencahayaan dari satu gambar.
* **Animasi:** menganimasikan wajah dengan deformasi kerutan yang realistis.
* **Kekokohan:** diuji pada gambar wajah dalam kondisi tidak terkontrol. Metode kami kokoh terhadap berbagai pose, pencahayaan, dan obstruksi.
* **Akurat:** rekonstruksi bentuk wajah 3D terkini pada dataset acuan [NoW Challenge](https://ringnet.is.tue.mpg.de/challenge).

## Memulai
Clone repositori:
  ```bash
  git clone https://github.com/Atyanta/3DFaceReconstruction/
  cd 3DFaceReconstruction

Python 3.7 (numpy, skimage, scipy, opencv)
PyTorch >= 1.6 (pytorch3d)
face-alignment (Opsional untuk mendeteksi wajah)
Anda dapat menjalankan

pip install -r requirements.txt
bash install_conda.sh
<!-- atau unduh data secara manual dari [FLAME 2020 model](https://flame.is.tue.mpg.de/download.php) dan [DECA trained model](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view?usp=sharing), dan letakkan di ./data -->


untuk memvisualisasikan prediksi landmark 2D, landmark 3D (merah berarti titik yang tidak terlihat), geometri kasar, geometri detail, dan kedalaman.
python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True
<p align="center"> <img src="Doc/images/id04657-PPHljWCZ53c-000565_inputs_inputs_vis.jpg"> </p> <p align="center"> <img src="Doc/images/IMG_0392_inputs_vis.jpg"> </p> Anda juga dapat menghasilkan file obj (dapat dibuka dengan Meshlab) yang mencakup tekstur yang diekstrak dari gambar input.

python demos/demo_transfer.py
Dengan gambar, Anda dapat merekonstruksi wajah 3D-nya, lalu menganimasikannya dengan mentransfer ekspresi dari gambar lain. Dengan menggunakan Meshlab untuk membuka file mesh obj detail, Anda dapat melihat sesuatu seperti ini:

<p align="center"> <img src="Doc/images/soubhik.gif"> </p> (Terima kasih kepada Soubhik yang mengizinkan saya menggunakan wajahnya ^_^)
Perhatikan, Anda perlu mengatur '--useTex True' untuk mendapatkan tekstur penuh.

python demos/demo_teaser.py 
DECA (kami) mencapai kesalahan rekonstruksi bentuk rata-rata 9% lebih rendah pada dataset NoW Challenge dibandingkan dengan metode terkini sebelumnya.
Gambar kiri membandingkan kesalahan kumulatif pendekatan kami dan metode terbaru lainnya (RingNet dan Deng et al. memiliki performa hampir identik, sehingga kurva mereka saling tumpang tindih). Di sini kami menggunakan jarak titik-ke-permukaan sebagai metrik kesalahan, mengikuti NoW Challenge.

<p align="left"> <img src="Doc/images/DECA_evaluation_github.png"> </p>
Untuk detail lebih lanjut mengenai evaluasi, silakan periksa makalah arXiv kami.

Pelatihan
Persiapkan Data Pelatihan

a. Unduh data gambar
Dalam DECA, kami menggunakan VGGFace2, BUPT-Balancedface, dan VoxCeleb2

b. Persiapkan label
FAN untuk memprediksi 68 landmark 2D
face_segmentation untuk mendapatkan masker kulit

c. Modifikasi dataloader
Dataloader untuk berbagai dataset ada di decalib/datasets, gunakan jalur yang tepat untuk gambar dan label yang sudah disiapkan.

Unduh model yang dilatih untuk pengenalan wajah
Kami menggunakan model dari VGGFace2-pytorch untuk menghitung kehilangan identitas, unduh resnet50_ft, dan letakkan di ./data

Mulai pelatihan

Latih dari awal:
python main_train.py --cfg configs/release_version/deca_pretrain.yml 
python main_train.py --cfg configs/release_version/deca_coarse.yml 
python main_train.py --cfg configs/release_version/deca_detail.yml

Dalam file yml, tuliskan jalur yang tepat untuk 'output_dir' dan 'pretrained_modelpath'.
Anda juga dapat menggunakan model yang dirilis sebagai model pra-latih, lalu abaikan langkah pra-pelatihan.

Kutipan

@inproceedings{DECA:Siggraph2021,
  title={Learning an Animatable Detailed {3D} Face Model from In-The-Wild Images},
  author={Feng, Yao and Feng, Haiwen dan Black, Michael J. dan Bolkart, Timo},
  journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)}, 
  volume = {40}, 
  number = {8}, 
  year = {2021}, 
  url = {https://doi.org/10.1145/3450626.3459936} 
}
