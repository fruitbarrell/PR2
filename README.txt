This is CISC442 Programing project.

In this project there are 4 folders:

--pycache
Compiled bytecode files of .py source files in this directory.
These compiled files are optimized versions of the code 
that Python uses to run faster the next time it executes the scripts.

--feature output
Folder in which the outputs of the feature_based stereo analysis are stored.


--region output
Folder in which the outputs of the region_based stereo analysis are stored


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

--RESULTS EXPLANATIONS
In total there is 20 results in this folder, 
11 for region_based and
9 for feature based.
All of them use a SEARCH_RANGE of 64 so that parameter
will be skipped in the explanation.
    ---Region based outputs
        For region based I used the following stereo pairs with the windows
            ----NCC outputs
                -Bull pair with a 7x7 template window 
                -Tsukuba pair with a 3x3 template window
                -Venus pair with a 11x11 template window
            ----SAD outputs
                -Bull pair with a 7x7 template window 
                -Tsukuba pair with a 3x3 template window
                -Venus pair with a 11x11 template window
                -sawtooth pair with a 5x5 template window
            ----SSD outputs
                -Bull pair with a 7x7 template window 
                -Tsukuba pair with a 3x3 template window
                -Tsukuba pair with a 7x7 template window
                -Venus pair with a 11x11 template window

    ---Feature based outputs
        For feature based I used the following stereo pairs with the windows
            ----NCC outputs
                -Poster Pair with a 5x5 template window
                -sawtooth Pair with a 3x3 template window
                -sawtooth Pair with a 7x7 template window
            ----SAD outputs
                -Poster Pair with a 5x5 template window
                -sawtooth Pair with a 3x3 template window
                -venus Pair with a 11x11 template window
            ----SSD outputs
                -Poster Pair with a 5x5 template window
                -sawtooth Pair with a 3x3 template window
                -Tsukuba Pair with a 7x7 template window

    ---Additional notes
    The functions are able to compute with a different SEARCH_RANGE
    other than 64, I just didn't change it so the difference between 
    scores is more visible. Same for the square template windows with
    odd sizes, they are able to be set to any size, I just like those sizes

                