
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "generated\examples\plot_bounding_box.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_generated_examples_plot_bounding_box.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_generated_examples_plot_bounding_box.py:


=================
Plot bounding box
=================

.. GENERATED FROM PYTHON SOURCE LINES 6-33



.. image-sg:: /generated/examples/images/sphx_glr_plot_bounding_box_001.png
   :alt: plot bounding box
   :srcset: /generated/examples/images/sphx_glr_plot_bounding_box_001.png
   :class: sphx-glr-single-img





.. code-block:: Python


    import numpy as np

    import vreg



    # Define geometry of source data
    input_shape = np.array([300, 250, 12])   # mm
    pixel_spacing = np.array([1.25, 1.25, 10]) # mm
    rotation_angle = 0.5 * (np.pi/2) # radians
    rotation_axis = [1,0,0]
    translation = np.array([0, -40, 180]) # mm
    margin = 10 # mm

    # Generate reference volume
    rotation = rotation_angle * np.array(rotation_axis)/np.linalg.norm(rotation_axis)
    input_affine = vreg.affine_matrix(rotation=rotation, translation=translation, pixel_spacing=pixel_spacing)
    #input_data, input_affine = vreg.fake.generate('double ellipsoid', shape=input_shape, affine=input_affine, markers=False)
    input_data, input_affine = vreg.generate('triple ellipsoid', shape=input_shape, affine=input_affine, markers=False)

    # Perform translation with reshaping
    output_data, output_affine = vreg.bounding_box(input_data, input_affine, margin=margin)

    # Display results
    vreg.plot_bounding_box(input_data, input_affine, output_data.shape, output_affine, off_screen=True)



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 1.819 seconds)


.. _sphx_glr_download_generated_examples_plot_bounding_box.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_bounding_box.ipynb <plot_bounding_box.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_bounding_box.py <plot_bounding_box.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_bounding_box.zip <plot_bounding_box.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
