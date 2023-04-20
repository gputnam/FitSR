#include  <algorithm>
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *shifted_sum(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
  {"shifted_sum", shifted_sum, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem =
{
  PyModuleDef_HEAD_INIT,
  "fitSR_chelpers",
  "",
  -1,
  module_methods
};


PyMODINIT_FUNC
PyInit_fitSR_chelpers(void) {
   // (void)Py_InitModule("fitSR_chelpers", module_methods);
   import_array();
   return PyModule_Create(&cModPyDem);
}

static PyObject *shifted_sum(PyObject *self, PyObject *args) { 

  // inputs
  PyObject *shift_obj;
  PyObject *path_obj;
  int noutput_time;

  if (!PyArg_ParseTuple(args, "OOi", &path_obj, &shift_obj, &noutput_time)) {
    return NULL;
  }

  PyObject *shift_arr = PyArray_FROM_OTF(shift_obj, NPY_INT64, NPY_IN_ARRAY);
  PyObject *path_arr = PyArray_FROM_OTF(path_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  if (shift_arr == NULL || path_arr == NULL) {
    Py_XDECREF(shift_arr);
    Py_XDECREF(path_arr);
    return NULL;
  }

  // Get the shapes of of everything
  //
  // There should be N angles and M pitches
  // Each input field response should have T current values, with T' current values outputed
  //
  // Then:
  // shift should be a (N x M) matrix
  // path should be a (M x T) matrix
  // output should be a (N x T') matrix
  int shift_ndim = (int) PyArray_NDIM((PyArrayObject*)shift_arr);
  int path_ndim = (int) PyArray_NDIM((PyArrayObject*)path_arr);

  if (shift_ndim != 2 || path_ndim != 2) {
    Py_XDECREF(shift_arr);
    Py_XDECREF(path_arr);
    return NULL;
  }

  npy_intp *shift_dim = PyArray_SHAPE((PyArrayObject*)shift_arr);
  int shift_N = shift_dim[0];
  int shift_M = shift_dim[1];

  npy_intp *path_dim = PyArray_SHAPE((PyArrayObject*)path_arr);
  int path_N = path_dim[0];
  int path_M = path_dim[1];

  if (shift_M != path_N) {
    Py_XDECREF(shift_arr);
    Py_XDECREF(path_arr);
    return NULL;
  }

  // zero-initialize output array
  npy_intp output_dim[2] {shift_N, noutput_time};
  PyObject *output = PyArray_SimpleNew(2, output_dim, NPY_DOUBLE); 
  if (output == NULL) {
    Py_XDECREF(shift_arr);
    Py_XDECREF(path_arr);
    return NULL;
  }
  PyArray_FILLWBYTE(output, 0);

  int t0_shift = (noutput_time - path_M) / 2;

  // The loop! Set output to the values in path, according to shift
  for (int i_shift = 0; i_shift < shift_N; i_shift++) {
    for (int i_path = 0; i_path < path_N; i_path++) {
      int s =  *((int*)PyArray_GETPTR2(shift_arr, i_shift, i_path)); 

      for (int tick_set = std::max(t0_shift+s, 0), tick_get=std::max(0, -t0_shift-s); 
           tick_get < path_M && tick_set < noutput_time; 
           tick_set++, tick_get++) {
       *((double*)PyArray_GETPTR2(output, i_shift, tick_set)) += *((double*)PyArray_GETPTR2(path_arr, i_path, tick_get));

      }
    }
  }

  Py_DECREF(shift_arr);
  Py_DECREF(path_arr);

  PyObject *ret = Py_BuildValue("O", output);
  return ret;

}


