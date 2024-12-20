
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "generated\examples\plot_affine_transform_reslice.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_generated_examples_plot_affine_transform_reslice.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_generated_examples_plot_affine_transform_reslice.py:


=============================
Plot affine transform reslice
=============================

.. GENERATED FROM PYTHON SOURCE LINES 6-47



.. image-sg:: /generated/examples/images/sphx_glr_plot_affine_transform_reslice_001.png
   :alt: plot affine transform reslice
   :srcset: /generated/examples/images/sphx_glr_plot_affine_transform_reslice_001.png
   :class: sphx-glr-single-img





.. code-block:: Python


    import numpy as np
    import vreg



    # Define geometry of input data
    input_shape = np.array([400, 300, 120])   # mm
    pixel_spacing = np.array([1.0, 1.0, 1.0]) # mm
    translation = np.array([0, 0, 0]) # mm
    rotation_angle = -0.2 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Define geometry of output data
    output_shape = np.array([350, 300, 10])
    pixel_spacing = np.array([1.25, 1.25, 5.0]) # mm
    translation = np.array([100, -30, -40]) # mm
    rotation_angle = 0.0 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    output_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)

    # Define affine transformation
    stretch = [1.25, 1, 1.0]
    translation = np.array([20, 0, 0]) # mm
    rotation_angle = 0.1 * (np.pi/2)
    rotation_axis = [1,0,0]
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    transformation = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=stretch)

    # Generate input data
    input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine)

    # Calculate affine transform
    output_data = vreg.affine_transform_and_reslice(input_data, input_affine, output_shape, output_affine, transformation)

    # Display results
    vreg.plot_affine_transform_reslice(input_data, input_affine, output_data, output_affine, transformation, off_screen=True)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 37.316 seconds)


.. _sphx_glr_download_generated_examples_plot_affine_transform_reslice.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_affine_transform_reslice.ipynb <plot_affine_transform_reslice.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_affine_transform_reslice.py <plot_affine_transform_reslice.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_affine_transform_reslice.zip <plot_affine_transform_reslice.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
