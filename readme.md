# 3D_Mag_Inv_Tess
An open-source, parallel C++ code for constrained 3D inversion magnetic data in spherical coordinates.  

Thanks for the below Important references:  
1. The framework of this constrained 3D inversion technique is based on a series of papers and researches by Dr. Yaoguo Li, Dr. Peter  Lelièvre, and Dr. Douglas Oldenburg (e.g., Li and Oldenburg, 1996, 1998, 2003; Lelièvre & Oldenburg, 2009).
2. [Eldar Baykiev](https://github.com/eldarbaykiev) provided an open-source algorithm [magnetic-tesseroids]( magnetic-tesseroids) (Baykiev et al., 2016), which we referred as the forward modelling part in this program.  
3. We followed some writing styles of the NGDC's [Geomagnetic Field Modeling software for the IGRF and WMM](https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html) .

<li><small>Baykiev, E., Ebbing, J., Brönner, M., & Fabian, K. (2016). Forward modeling magnetic fields of induced and remanent magnetization in the lithosphere using tesseroids. Computers & Geosciences, 96, 124-135. doi: 10.1016/j.cageo.2016.08.004</small></li>

<li><small>Lelièvre, P. G., & Oldenburg, D. W. (2009). A comprehensive study of including structural orientation information in geophysical inversions. Geophysical Journal International, 178, 623-637. doi: 10.1111/j.1365-246X.2009.04188.x</small></li>

<li><small>Li, Y., & Oldenburg, D. W. (1996). 3-D inversion of magnetic data. Geophysics, 61(2), 394-408. doi: 10.1190/1.1443968</small></li>

<li><small>Li, Y., & Oldenburg, D. W. (1998). 3-D inversion of gravity data. Geophysics, 63(1), 109-119. doi: 10.1190/1.1887478</small></li>

<li><small>Li, Y., & Oldenburg, D. (2003). Fast inversion of large‐scale magnetic data using wavelet transforms and a logarithmic barrier method. Geophysical Journal International, 152, 251-265. doi: 10.1046/j.1365-246X.2003.01766.x</small></li>

# Compilation Instructions
## Required software packages
- Intel c++ compiler: icpc >= 2021.3.0
- Intel one Math Kernel Library (>=2021.3)

## Compilation in Linux system or Windows Subsystem for Linux
```sh
icpc IversionMagnetic_tess_3D_V12_cmdline.cpp -o 3D_Mag_Inv_Tess.o -mkl=parallel -qopenmp -std=c++11
```

# User Guides
## Command Line 
1. For help
	```sh
	./3D_Mag_Inv_Tess.o h
	```

2. Using a parameter file (recommanded)
	```sh
	./3D_Mag_Inv_Tess.o f parameter_file
	```
	The parameter file's content can refer to the "para_example.dat", which includes information about the magnetic_anomaly_file, mesh_file, and parameters used for inversion.

3. Using a simple cmd line
	```sh
	./3D_Mag_Inv_Tess.o mag_anomaly_file mesh_file regu_para
	```
	Only the magnetic anomaly file, mesh file, and the used regularization parameter are used in this simple inversion, other parameters are all default values or settings.

## Parameter file
```
-----Please_input_the_observed_data_file:
magnetic_anomaly_filename.extention
-----Please_input_the_[mesh]_file:
mesh_filename.extention
-----Please_input_the_[alpha-s_-x_-y_-z]:
1 1 1 1
-----Please_input_the_parameter_whether_U_need_[GCV]_to_calculate
-----the_best_regularization_parameter_or_not:
1
-----Please_input_the_[TYPE]_of_the_depth_weighting_function(1_for_1998_usual_weighting_and_2_for_2000_based_on_sensitivity)
-----and_[uniform_altitude(z0_upward_positive)]_for_1:
2
5000
-----Please_input_the_[beta_value]_of_the_depth_weighting_function_[default_value]_[3_for_1]_and_[1_for_2]
-----and_[regularization_parameter]_for_THIS_inversion:
1
5.8
-----Please_input_the_parameter_which_bound_constrained_method_you_choosed
2
-----Please_input_the_[_reference_model]_[origin_model]_[minimum_model]_[maximum_model]
0 0.001 0 10
-----Please_input_the_[parameter]_whether_using_a_spatial_weighting_function_and_[the_file's_name]:
1
SpatialWeight_forInv_20201119.xyz
-----Please_input_the_parameter_whether_using_a_[non_uniform_reference_model]_and_[the_file's_name]:
0
ref_data.dat
-----Please_input_the_parameter_whether_using_a_[non_uniform_origin_model]_and_[the_file's_name]:
0
sus_ori.dat
-----Please_input_the_parameter_whether_using_a_[non_uniform_minimum_model]_and_[the_file's_name]:
0
sus_min.dat
-----Please_input_the_parameter_whether_using_a_[non_uniform_maximum_model]_and_[the_file's_name]:
0
sus_max.dat
```
**Notice**: all the lines with "-----" as beginings can not be deleted or add new lines.

## Anomaly file
Eight columns are needed.
(**coor_lon**, **coor_lat**,  **coor_alt**) are the coordinates of the observation position of the **total-field_anomaly** whose error is **standard_deviation**.  
(**IGRF_x**, **IGRF_y**, **IGRF_z**) are the three components of the reference field which can be obtained using IGRF software.  

```
coor_lon coor_lat coor_alt IGRF_x IGRF_y IGRF_z total-field_anomaly standard_deviation
```
For example,  
```
83.05 42.05 5000 27159.7 1042.7 47821.1 -3.86481238 1
```  

## Mesh file
Eleven columns are needed.   
**lon_min** : western longitude border in degrees  
**lon_max** : eastern longitude border in degree  
**lat_min** : southern latitude border in degrees  
**lat_max** : northern latitude border in degrees  
**top** : top depth  
**bot** :  bottom depth  
**density** : Unit value (=1000 kg/m3)  
**susceptibility** : Unit value (=1)  
**IGRF_x** : x-component of ambient magnetic field  
**IGRF_y** : y-component of ambient magnetic field  
**IGRF_z** : z-component of ambient magnetic field   

For example,
```
83 83.1 42 42.1 0 -5000 1000 1 23100 1300 53000
```  

**Notice**: The above coordinates in both the anomaly and the mesh file are longitudes and latitudes in a geocentric spherical coordinates system, and the altitudes are relative to the Earth's mean radius.


## Spatial weighting functions
Seven columns are needed.   
```
w_s w_x diff_x w_y diff_y w_z diff_z
```

## Reference/Initial/Minimum/Maximum model
Single column file of which row number equals to the number of the cells in the mesh file.

# Example
A folder named "Inv_example" includes parameter file, anomaly file, mesh file, and spatial weighting functions file.  
The number of the observation positions and the cells in mesh file are 3500 and 39287, respectively.  
Due to the sensitivity kernel matrices have the largest memory demand, before running an inversion test we can just estimate its demand.   
For example, 3500 × 39287 × 8 byte ≈ 1.025 Gb;  
In the present version of this software, we need four sensitivity kernel matrices with same memory demands, that are related to the three components of the anomalous vector and the total-field anomaly.  
Thus, for this example the total memory demand of four large matrices should be larger than 1.025 × 4 ≈ 4.1 Gb.  
Considering there are other smaller matrices and many temp vectors that also need space, so a hardware with memory larger than 5 Gb is recommended for running this example.


# Contact
Shida Sun (shidasun.cug@gmail.com, sdsun@hgu.edu.cn)
