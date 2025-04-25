This is CISC442 Programing project.

In this project there are 4 folders:

--pycache
Compiled bytecode files of .py source files in this directory.
These compiled files are optimized versions of the code 
that Python uses to run faster the next time it executes the scripts.

--feature output
Folder in which the outputs of the feature_based stereo analysis are stored.
    --NCC folder
    Only one output is present, obtained by
    using the sawtooth pair, a 64 search range and 7x7 matching window
    --SAD folder
    Only one output is present, obtained by
    using the venus pair, a 64 search range and 11x11 matching window
    --SSD folder
    Only one output is present, obtained by
    using the Tsukuba pair, a 64 search range and 7x7 matching window

--region output
Folder in which the outputs of the region_based stereo analysis are stored
    --NCC folder
    Only one output is present, obtained by
    using the venus pair, a 64 search range and 11x11 matching window
    --SAD folder
    Only one output is present, obtained by
    using the sawtooth pair, a 64 search range and 5x5 matching window
    --SSD folder
    Only one output is present, obtained by
    using the Tsukuba pair, a 64 search range and 7x7 matching window

--TEST images
Folder in which the testing stereo pairs are stored. There a 3 total pairs:
Tsukuba-sawtooth-Venus

In addition to the folders, there also are 5 scripts with various function in them.
In this file the overall purpose of each script is explained,
if you wish to know what each function does refer to its file
and see the comment under its definition

--create.py
A script intended to work as the interface that the user interacts with to generate the desired stereo analysis.
The User just has to run the file and follow the instructions displayed on the terminal.

--feature.py
Script containing all funtions needed to perform feature based stereo analysis.
3 functions total.
Uses Multiproccesing to reduce loading times.

--region.py
Script containing all funtions needed to perform region based stereo analysis.
1 function total.

--score_funtions.py
Contains the functions used to get matching scores to perform the analysis.
3 functions total.(SSD,SAD,NCC)
These functions additionally used a normalized version of the inputted patch.

--validation_functions.py
This script contains 2 funtions that are used at the last steps of the analysis,
left-right consistency check and fill disparity gaps adaptive, 
which a fancy name for the function that average the neighbourhood of a pixel.
