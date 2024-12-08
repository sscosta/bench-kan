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
        <li><a href="#docker">Docker</a></li>
        <li><a href="#locally">Locally</a></li>
      </ul>
    </li>
    <li>
      <a href="#testing-with-own-dataset">Testing with own dataset</a>
      <ul>
        <li>
          <a href="#basic-structure">Basic Structure</a>
          <ul>
            <li><a href="#workflow">Workflow</a></li>
            <li><a href="#stress-test">Stress Test</a></li>
            <li><a href="#test">Test</a></li>
            <li><a href="#dictionary-file">Dictionary File</a></li>
            <li><a href="#verifications">Verifications</a></li>
            <li><a href="#retain">Retain</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<img src="images/screenshot.png" alt="Logo">

Bench-KAN is a repository for validation of Kolmogorov-Arnold Networks (KANs) for real-world applications, as well as for the usage of KANs for real-time Human Activity Recognition (HAR). KANs are based on the generalization of the Kolmogorov-Arnold Representation Theorem (KAT) for networks of arbitrary depth and width. KAT states that any continuouos multivariate function may be composed by the sum of univariate continuous functions.
KANs were validated across mathematical and physics domains but no broad study on their usage for real-world problems had been done. The present project performs a study on open datasets with a focus on improving the performance of Human Activity Recognition Systems.

The repository is composed of code to test KANs with open datasets and of two android applications, one for data collection (CollectData) and another for real-time human activity recognition (HARDetector).

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
3. Run the benchmark script that trains and tests KANs on all datasets<br/>
   ```sh
   python bench-exec.py
   ```
4. Collect the tensorflow lite model for the dataset har mobile<br/>

5. Connect an Android device into the computer.
   Plug an Android device into the computer and copy the tensorflow lite model to the external storage. Create a folder named SensorData and copy the tensorflow lite model to the folder.

6. Open the HARDetector project on Android Studio<br/>
  Compile the project. Install the resulting application in the plugged Android device.
  
7. Place the Android Device on the waist and enjoy!
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
As mentioned in the introduction, this app takes advantage of <i>TSL</i> files. 
</p>

<p>These are <i>YAML</i> files with the purpose of defining specific tests based on HTTP requests in order to specify tests that couldn't reliably be made automaticly with just the API's specification.</p> 

The app supports the creation of these files trough a simple UI, however not every functionality is supported trough this UI, leaving some functionalities to only a manual creation. 

### Basic Structure 

You can write _TSL_ files in _YAML_. A sample _TSL_ definition written in _YAML_ looks like:
```sh
  - WorkflowID: crud_pet

    Stress:
      Count: 40
      Threads: 5
      Delay: 0

    Tests:

    - TestID: createPet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet"
      Method: Post
      Headers:
        - Content-Type:application/json
        - Accept:application/json
      Body: "$ref/dictionary/petExample"
      Retain:
        - petId#$.id
      Verifications:
        - Code: 200
          Schema: "$ref/definitions/Pet"

    - TestID: readPet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet/{petId}"
      Method: Get
      Headers:
        - Accept:application/xml
      Verifications:
        - Code: 200
          Schema: "$ref/definitions/Pet"
          Custom: ["CustomVerification.dll"]

    - TestID: updatePet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet/{petId}"
      Query:
        - name=doggie
        - status=sold
      Method: Post
      Headers:
        - Accept:application/xml
      Verifications:
        - Code: 200
          Schema: "$ref/dictionary/petSchemaXml"

    - TestID: deletePet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet/{petId}"
      Method: Delete
      Headers:
        - Accept:application/json
      Verifications:
        - Code: 200     
```
#### Workflow

Every _TSL_ file must include atleast one workflow 
```sh
  - WorkflowID: crud_pet
```
The workflow needs an ID, which must be unique across all workflows. <br/>
One workflow must be comprised of one or more tests and optionally one stress test. All the tests inside the workflow are guaranteed to be executed sequentially which is usefull for situations where the output of one test influences the input of the other.

#### Stress Test

Optionally, one workflow can have one stress test 
```sh
  Stress:
      Count: 40
      Threads: 5
      Delay: 0
```
The stress test has 3 fields, Count, Threads and Delay.<br/>
* Count defines the number of times the complete workflow will be executed
* Threads defines the number of threads by which _Count_ will be divided (Count: 10 and Threads: 2 means each thread will execute the workflow 5 times)
* Delay defines the delay in milliseconds between every full execution, which can be usefull to prevent errors of too many executions 

#### Test

You can define one or more tests within each workflow
```sh
  Tests:

    - TestID: createPet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet"
      Method: Post
      Headers:
        - Content-Type:application/json
        - Accept:application/json
      Body: "$ref/dictionary/petExample"
      Retain:
        - petId#$.id
      Verifications:
        - Code: 200
          Schema: "$ref/definitions/Pet"

    - TestID: readPet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet/{petId}"
      Method: Get
      Headers:
        - Accept:application/xml
      Verifications:
        - Code: 200
          Contains: id
          Count: doggie#1
          Schema: "$ref/definitions/Pet"
          Match: /Pet/name#doggie
          Custom: ["CustomVerification.dll"]

    - TestID: updatePet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet/{petId}"
      Query:
        - name=doggie
        - status=sold
      Method: Post
      Headers:
        - Accept:application/xml
      Verifications:
        - Code: 200
          Schema: "$ref/dictionary/petSchemaXml"

    - TestID: deletePet
      Server: "https://petstore3.swagger.io/api/v3"
      Path: "/pet/{petId}"
      Method: Delete
      Headers:
        - Accept:application/json
      Verifications:
        - Code: 200     
```

The test is identified with its id, which MUST be unique across all tests of every workflow
```sh
   - TestID: createPet
```

It is then followed by 3 mandatory fields for a successful HTTP request, the server, the path and the method
```sh
    Server: "https://petstore3.swagger.io/api/v3"
    Path: "/pet"
    Method: Post
```
(Currently the only supported methods are Get, Post, Put and Delete)

You can also define headers following a key/value structure
```sh
  Headers:
    - Content-Type:application/json
    - Accept:application/json
```

Which is similar for Query string parameters
```sh
  Query:
    - name=doggie
    - status=sold
```

The body data can be defined directly on the _TSL_ file, however they can sometimes be extremely large which would hurt the clarity and readability of the file. You can however create an auxiliary text file (dictionary file) which contains the actual body and a unique identifier within the dictionary, leaving only the reference to said identifier on the _TSL_ file
```sh
  Body: "$ref/dictionary/petExample"
```

#### Dictionary File

The dictionary file is a text file containing all the body data and schemas in order to improve clarity on the actual _TSL_ file
```sh
  dictionaryID:petExample
  {
    "id": 10,
    "name": "doggie",
    "status": "available"
  }

```
Every entry on the file requires the dictionaryID which must be unique across all entries, followed by the actual data, followed by an empty line to separate them

#### Verifications

Each test can have multiple verifications, only the Code verification is mandatory
```sh
  Verifications:
    - Code: 200
      Contains: id
      Count: doggie#1
      Schema: "$ref/definitions/Pet"
      Match: /Pet/name#doggie
      Custom: ["CustomVerification.dll"]
```
Currently 6 different verifications are supported:

| Requirement 	| Name     	| Input Type        	| Description                                                     	|
|-------------	|----------	|-------------------	|-----------------------------------------------------------------	|
| Mandatory   	| Code     	| Integer           	| Response code matches the given code                            	|
| Optional    	| Contains 	| String            	| Response body contains the given string                         	|
| Optional    	| Count    	| String#Integer    	| Response body contains given string # times                     	|
| Optional    	| Schema   	| String            	| Response body matches the given schema*                          	|
| Optional    	| Match    	| StringPath#String 	| Response body matches the given value present in the StringPath 	|
| Optional    	| Custom   	| [String]          	| Runs Custom verifications given by the user                     	|

\*The schema verification can be supplied directly, or through reference to the dictionary file or to any schema present in the supplied OAS  

#### Retain

In some requests, the input is based on the output of a previous request, usually in simple workflows, like create read<br/>
```sh
    Retain:
      - petId#$.id
```
The keyword Retain allows the user to retain some information from the response body of the request to then be used in other tests of the same workflow.<br/>
For instance the value present at the json path _$.id_ will be retained with the identifier _petId_ which can then be used in following tests
```sh
    Path: "/pet/{petId}"
```

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