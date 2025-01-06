"""
========================
Plot affine with reslice
========================
"""
import time
import vreg

n=4

if n==1:
    input_data, input_affine, output_affine, output_shape = vreg.generate_reslice_data_1()
elif n==2:
    input_data, input_affine, output_affine, output_shape = vreg.generate_reslice_data_2()
elif n==3:
    input_data, input_affine, output_affine, output_shape = vreg.generate_reslice_data_3()
elif n==4:
    input_data, input_affine, output_affine, output_shape = vreg.generate_reslice_data_4()
elif n==5:
    input_data, input_affine, output_affine, output_shape = vreg.generate_reslice_data_5()
elif n==6:
    input_data, input_affine, output_affine, output_shape = vreg.generate_reslice_data_6()

start_time = time.time()
output_data, output_affine = vreg.affine_reslice(input_data, input_affine, output_affine, output_shape=output_shape)
end_time = time.time()

# Display results
print('Computation time (sec): ', end_time-start_time)
pl = vreg.plot_affine_resliced(input_data, input_affine, output_data, output_affine, off_screen=True)
pl.show_grid()
pl.show()

