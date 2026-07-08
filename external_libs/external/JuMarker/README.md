
# JuMarker
**Authors:** [David Jurado-Rodríguez](mailto:ep2jurod@uco.es), [Rafael Muñoz-Salinas](mailto:rmsalinas@uco.es), [Sergio Garrido-Jurado](mailto:sgj@seaberyat.com) and [Rafael Medina-Carnicer](mailto:rmedina@uco.es) 

### [Overview]

JuMarker is library to design, detect and track customized fiducial markers as an alternative to the traditional squared ones. In industrial environments the visual appearance is a crucial aspect, as we know the traditional black and white markers present unattractive appearance. So, in contrast customized markers presented in this work provide attractive designs which can be integrated naturally in industrial and commercial environments.


### [How to cite]
If you use JuMarker in an academic work, please cite:

D. Jurado-Rodríguez, R. Muñoz-Salinas, S. Garrido-Jurado and R. Medina-Carnicer, **Design, Detection, and Tracking of Customized Fiducial Markers**, in _IEEE Access_, vol. 9, pp. 140066-140078, 2021, doi: 10.1109/ACCESS.2021.3118049. **[PDF](https://ieeexplore.ieee.org/document/9559993)**.

    @article{jurado2021,
      title={Design, Detection, and Tracking of Customized Fiducial Markers},
      author={Jurado-Rodríguez, David and Muñoz-Salinas, Rafael and Garrido-Jurado, Sergio and Medina-Carnicer, Rafael},
      journal={IEEE Access},
      volume={9},
      pages={140066-140078},
      doi = {10.1109/ACCESS.2021.3118049},
      year={2021}
     }


### [Prerequisites]
We have tested JuMarker in Ubuntu 18.04,  and 20.04, but it should be easy to compile in other platforms.
The only required dependency is OpenCV, at least v4.5.


### [Compilation]
JuMarker can be compiled using cmake:
```
mkdir build
cd build
cmake .. -DOPENCV_PATH=/path/to/opencv/installation
make
```

### [How to use]
First you have to create the customized marker design using a vecor graphics software such as Inkscape.
See some examples in the *marker_designs* folder.

To create the images of the customized markers:
```
./create_marker [svg file] [output folder] [markers number]
```

To test the detection of the customized marker:
```
./jumarker_test [svg file] [total id bits] -c [cameraCalibration file] -v [input video]
```

An example of camera calibration file can be found in *utils* folder.


### [License]
JuMarker is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
Public License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
JuMarker is distributed in the hope that it will be useful for non-commercial academic research, but without any warranty.
Please see *LICENSE.txt* for more details.


### [Acknowledgments]
This project has been funded under the Industrial Phd Program of University of Córdoba with Seabery R\&D and Project TIN2019-75279-P of Spanish Ministry of Economy, Industry and Competitiveness, and FEDER.
