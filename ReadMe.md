#DeTurbulence non-Deep implementation

This is a simple implementation of DeTurbulence method using image processing techniques.

The algorithms is as follows:
1) A ROI is chosen in the image.
2) Reference image is created/chosen.
3) Registration to the reference image using TV-L1 optiacal flow.
4) Iterative fusion of registered frames.
5) Frame Deconvolution for diffraction limited PSF.

