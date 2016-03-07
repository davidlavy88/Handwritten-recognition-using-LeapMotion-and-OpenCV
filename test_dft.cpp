#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <ctype.h>
#include <iomanip>
#include <math.h>
#include <complex>
#include <numeric>
#include <cfloat>

using namespace std;

vector<vector<float>> ExtractDFTCoeff(vector<vector<float>> &new_character)
{
//    vector<complex<float>> complex_shape, complex_dft;
//    for (int i=0;i<new_character.size();i++)
//    {
//        complex<float> xiy(new_character[i][0],new_character[i][1]);
//        complex_shape.push_back(xiy);
//    }
    int m(new_character.size());
    int i,k;
    float arg, cosarg, sinarg;
    float tempx, tempy;
//    vector<float> x2,y2;
    vector<float> cmplx;
    vector<vector<float>> dft;
//    x2.reserve(new_character[0].size());
//    y2.reserve(new_character[0].size());
    for(i=0;i<m;i++)
    {
	cmplx.clear();
        tempx = 0;
        tempy = 0;
        arg = -1 * 2.0 * 3.141592654 * (float)i / (float)m;
        for (k=0;k<m;k++)
        {
            cosarg = cos(k*arg);
            sinarg = sin(k*arg);
            tempx += (new_character[k][0] * cosarg - new_character[k][1] * sinarg);
            tempy += (new_character[k][0] * sinarg + new_character[k][1] * cosarg);
        }
	cmplx.push_back(tempx); cmplx.push_back(tempy); 
	dft.push_back(cmplx);
        //x2.push_back(tempx/(float)m);
        //y2.push_back(tempy/(float)m);
    }
    return dft;
}

int main()
{
    string input;
    vector<string> all;
    vector<vector<vector<float>>> oldLetters, newLetters;
    vector<vector<float>> data, iniQueryVector, test;
    vector<float> query;
    ifstream infile;
    int finish = 0;
    while (finish != 1)
    {
        cout << "Enter the initial 2 or 3 letters of the file to read: ";
        cin >> input;
        if (input == "none")
        {
            finish=1;
            break;
        }
        else
        {
            data.clear(); query.clear();
            float val = 0;
            infile.open(input+"_2d.txt");
            all.push_back(input+"_2d.txt");
            string   line;

            while(getline(infile, line))
            {
                vector<float>   lineData;
                stringstream  linestream(line);

                float value;
                int count(1);
                // Read an integer at a time from the line
                while(linestream >> value)
                {
                    if (count != 1)
                    {
                        // Add the integers from a line to a 1D array (vector)
                        lineData.push_back(value);
                    }
                    count++;
                }
                // When all the integers have been read add the 1D array
                // into a 2D array (as one line in the 2D array)
                data.push_back(lineData);
                query.push_back(val);
                val++;
            }
            cout << data.size() << endl;
            iniQueryVector.push_back(query);
            oldLetters.push_back(data);
            infile.close();
            continue;
        }
    }
    cout << oldLetters.size() << endl;
    for (int i=0;i<oldLetters.size();i++)
    {
	test = ExtractDFTCoeff(oldLetters[i]);
    }
    cout << "This should be 256: " << test.size() << endl;
    cout << "Values at i=37 : " << test[36][0] << "," << test[36][1] << endl;
    return 0;
}
