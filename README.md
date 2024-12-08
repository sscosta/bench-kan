<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sscosta/bench-kan">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Bench-KAN</h3>

  <p align="center">
    Real-time Human Activity Recognition with Bench-KAN
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#locally">Locally</a></li>
        <ul>
          <li><a href="#prerequisites">Prerequisites</a></li>
          <li><a href="#installation">Installation</a></li>
        </ul>
        <li><a href="#run-the-benchmark-and-do-real-time-har">Run the Benchmark and Do Real-time HAR</a></li>
      </ul>
    </li>
    <li>
      <a href="#testing-with-own-dataset">Testing with Own Dataset</a>
      <ul>
        <li><a href="#collecting-data-with-activity-protocol">Collecting Data with Activity Protocol</a></li>
        <li><a href="#collect-data-for-an-isolated-activity">Collect Data for an Isolated Activity</a></li>
        <li><a href="#produce-a-train-and-test-dataset">Produce a Train and Test Dataset</a></li>
        <li><a href="#train-the-model-with-your-dataset">Train the Model with Your Dataset</a></li>
        <li><a href="#assessing-the-performance-of-the-models">Assessing the Performance of the Models</a></li>
        <li><a href="#load-the-trained-model-in-the-device">Load the Trained Model in the Device</a></li>
        <li><a href="#validation">Validation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact-and-acknowledgements">Contact and Acknowledgements</a></li>
  </ol>
</details>




<!-- ABOUT THE PROJECT -->
## About The Project

<img src="images/screenshot.png" alt="Logo">

Bench-KAN is a repository for validation of Kolmogorov-Arnold Networks (KANs) for real-world applications, as well as for the usage of KANs for real-time Human Activity Recognition (HAR). KANs are based on the generalization of the Kolmogorov-Arnold Representation Theorem (KAT) for networks of arbitrary depth and width. KAT states that any continuouos multivariate function may be composed by the sum of univariate continuous functions.
KANs were validated across mathematical and physics domains but no broad study on their usage for real-world problems had been done. The present project performs a study on open datasets with a focus on improving the performance of Human Activity Recognition Systems.

The repository is composed of code to test KANs with open datasets and of two android applications, one for data collection (CollectData) and another for real-time human activity recognition (HARDetector).

The real-time component can be used with a dataset comprised of 6 subjects. It can also be used with your own dataset.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

The benchmark source code needs the following python dependencies:

* [pytorch](https://pytorch.org/)
* [tqdm](https://tqdm.github.io/)
* [sci-kit learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)

The android applications CollectData and HARDetector run with built-in Android and [Kotlin](https://kotlinlang.org/) dependencies, as well as the [pytorch](https://pytorch.org/) java binding.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Locally

Here are the steps to run the scripts and app.

#### Prerequisites

* Install Android Studio
* Install Python (The current scripts were tested with Python 3.10 for pytorch compatibility)

#### Installation

After having installed all the required software:

1. Clone the repo
   ```sh
   git clone https://github.com/sscosta/bench-kan
   ```
2. Install python dependencies

    ``` sh
    cd bench-kan
    pip install -r requirements.txt
    ```
#### Run the benchmark and do real-time HAR
1. Run the benchmark script that trains and tests KANs on all datasets<br/>
   ```sh
   python bench-exec.py
   ```
2. Collect the tensorflow lite model for the dataset har mobile<br/>

3. Connect an Android device into the computer.
   Plug an Android device into the computer and copy the tensorflow lite model to the external storage. Create a folder named SensorData and copy the tensorflow lite model to the folder.

4. Open the HARDetector project on Android Studio<br/>
  Compile the project. Install the resulting application in the plugged Android device.
  
5. Place the Android Device on the waist and enjoy!
    The app correcly identify which activity you're performing<br/>
    The system is prepared to identify the following activities:
    * LAYING
    * SITTING
    * STANDING
    * WALKING
    * WALKING UPSTAIRS
    * WALKING DOWNSTAIRS

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Testing with own dataset

<p>

Apart from using the har_mobile dataset, you can build your own dataset, train a model with it and test the system.
</p>


<p>

To build your own dataset, open the CollectData project on Android Studio and compile the project. Install it in an Android phone and start the application. You can either collect data by performing an activity protocol or collect data for an isolated activity.
</p>

### Collecting data with Activity protocol 

Enter the Subject Name and press Start Activity Protocol.

The system will instruct you to perform a protocol of 6 activities, each during 30 seconds with 5 secondsinterval between them.

You will start by lay down, then sit, stand, walk, walk upstairs and downstairs.

The application will collect data with 50Hz sampling frequency and annotate it with the subject name.


### Collect data for an isolated activity

If you don't want to perform the activity protocol and just want to collect data for one of the activities, select the activity from the activity drop down menu, and press Start <Activity>. When you're ready to stop collecting data, press Stop <Activity>.


### Produce a train and test dataset

The datasets for each subject will be stored in the external storage of the device, namely in the SensorData folder. The files will be marked with the subject name and a timestamp.

Collect those files and combine them in a file called final_combined_output.csv.

Run the preprocess script and perform a train/test split of the dataset:
```sh
python preprocess.py
python train_test_split.py
```

### Train the model with your dataset

Place the resulting files (har_mobile_train.csv and har_mobile_test.csv) in the folder preprocessed_data.
Run the training script:
```sh
python bench-exec.py
```
From the kan_results folder, select the kan model you want to load in the Android device (there will be 7, one for each of the kernels). Those files end with the suffix _lite.pt.

### Assessing the performance of the models

You can use the script generate_plot.py to plot the results of each individual kernel for your dataset. You can also lookup accuracy, precision, recall and f1-score for each of the training epochs to see if and how the model converged.

### Load the trained model in the device

Copy the trained model to the SensorData folder of the device and rename it KAN.pt.

### Validation

After you load the trained model in the device, place it in the waist and see it perform real-time human activity recognition.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact and Acknowledgements

This work is supported by NOVA LINCS Research Unit, ref. UIDB/04516/2020 (https://doi.org/10.54499/UIDB/04516/2020), and the LASIGE Research Unit, ref. UIDB/00408/2020 (https://doi.org/10.54499/UIDB/00408/2020) and ref. UIDP/00408/2020 (https://doi.org/10.54499/UIDP/00408/2020) with financial support from [FCT](https://www.fct.pt/en/), through national funds. We gratefully acknowledge [Celfocus](https://www.celfocus.com/) for their support and generosity with allocated time for this investigation. We also thankfully acknowledge Lisbon School of Engineering ([ISEL](https://www.isel.pt/)) for openhandedly providing access to hardware infrastructure essentialfor conducting the experiments.

Samuel Costa - A43552@alunos.isel.pt</br>
Matilde Pato - matilde.pato@isel.pt</br>
Nuno Datia - nuno.datia@isel.pt</br>

Project Link: [https://github.com/sscosta/bench-kan](https://github.com/sscosta/bench-kan)

</br>To cite the work, please use the following text:</br>
Costa, S.; Pato, M.; Datia, N.</br>
An empirical study on the application of KANs for classification </br>
ICAAI'24 - The 8th International Conference on Advances in Artificial Intelligence, 2024

<p align="right">(<a href="#top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png