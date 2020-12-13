#DeTurbulence non-Deep implementation

This is a simple implementation of DeTurbulence method using image processing techniques.

The algorithms is as follows:
0) A ROI is chosen in the image.
1) Reference image is created/chosen.
2) Registration to the reference image using TV-L1 optiacal flow.
3) Iterative fusion of registered frames.
4) Frame Deconvolution for diffraction limited PSF.

