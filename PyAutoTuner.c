/* Copyright 2012, Eng Eder de Souza 

   This program is free software; you can redistribute it and/or modify        
   it under the terms of the GNU General Public License as published by        
   the Free Software Foundation; either version 2 of the License, or           
   (at your option) any later version.                                         
                                                                                
   This program is distributed in the hope that it will be useful,             
   but WITHOUT ANY WARRANTY; without even the implied warranty of              
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               
   GNU General Public License for more details.                                
                                                                                
   You should have received a copy of the GNU General Public License           
   along with this program; if not, write to the Free Software                 
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  

*/


#include <Python.h>
#include <numpy/arrayobject.h>
#include "mayer_fft.h"
#include "autotalent.h"


static PyObject *Tuner(PyObject *self, PyObject *args)

{

	float *buffer, concert_a, fixed_pitch, fixed_pull, corr_str, corr_smooth, pitch_shift, lfo_depth, lfo_rate, lfo_shape, lfo_symm, form_warp, mix;
	char *key;
	int fs, FrameSize, scale_rotate, lfo_quant, form_corr;
	PyArrayObject *arr;
	PyObject *In_object;



	if (!PyArg_ParseTuple(args, "Oiiiiiffffffffffffc", &In_object,&fs,&FrameSize,&scale_rotate,&lfo_quant,&form_corr,&concert_a,&fixed_pitch,&fixed_pull,&corr_str,&corr_smooth,&pitch_shift,&lfo_depth,&lfo_rate,&lfo_shape,&lfo_symm,&form_warp,&mix,&key))
		return NULL;

	npy_intp ArrLen[1]={FrameSize};

	Py_BEGIN_ALLOW_THREADS;

	instantiateAutotalentInstance(fs);

	PyArrayObject *x_array = PyArray_FROM_OTF(In_object, NPY_FLOAT, NPY_IN_ARRAY);
	if (x_array == NULL) {
    		Py_XDECREF(x_array);
    		return NULL;
	}

	buffer = (float*)PyArray_DATA(x_array);

	initializeAutotalent(&concert_a, &key, &fixed_pitch, &fixed_pull, &corr_str, &corr_smooth, &pitch_shift, &scale_rotate, &lfo_depth, &lfo_rate, &lfo_shape, &lfo_symm, &lfo_quant, &form_corr, &form_warp, &mix);

	processSamples(buffer, FrameSize);

	arr = (PyArrayObject *)PyArray_SimpleNewFromData(1, ArrLen,NPY_FLOAT,buffer);

	Py_END_ALLOW_THREADS;

	return PyArray_Return(arr);

	*buffer = 0;

	freeAutotalentInstance();


}

static PyMethodDef Tuner_methods[] = { 
    	{   
        	"Tuner", Tuner, METH_VARARGS,
        	"Print 'Tuner Crazy' from a method defined in a C extension."
    	},  
	{NULL, NULL, 0, NULL}
};


static struct PyModuleDef Tuner_definition = { 
	PyModuleDef_HEAD_INIT,
	"Tuner",
	"A Python module from C code.",
	-1, 
	Tuner_methods
};


PyMODINIT_FUNC PyInit_AutoTune(void) {
	Py_Initialize();
	import_array();
	return PyModule_Create(&Tuner_definition);
}
