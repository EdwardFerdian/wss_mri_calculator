import logging
import wss_utils
import numpy as np
import matplotlib.pyplot as plt

def test_vector_magnitude():
    v1 = np.asarray([[0.370491361,	-0.167037462,	-0.225748788],
                    [0.585235848,	-0.099175084,	-0.116312987]])
    v1_mag = wss_utils.get_vector_magnitude(v1)
    
    true_mag = [0.464895555, 0.60486809]
    print('calculated mag', v1_mag)
    print('actual mag', v1_mag)
    np.testing.assert_almost_equal(v1_mag, true_mag)

    print('test_vector_magnitude PASS')

def test_orthogonal_vectors():
    v1 = np.asarray([[0.64604717, 0.0196206, 0.08837089],
        [0.64604717, 0.0196206, 0.08837089]])
    normals = np.asarray([[0.6021105,	0.40786213,	0.6863755],
        [0.24887183,    0.48617426,	0.8376738]])

    normal_vectors, tangent_vectors = wss_utils.get_orthogonal_vectors(v1, normals)

    # Actual values
    true_normal = np.asarray([
        [0.275555809, 0.186658062, 0.314119678],
        [0.060811322, 0.118795684, 0.204683877]
    ])
    true_tangent = np.asarray([
        [0.370491361, -0.167037462, -0.225748788],
        [0.585235848, -0.099175084, -0.116312987]
    ])

    # Assert
    np.testing.assert_almost_equal(normal_vectors, true_normal)
    np.testing.assert_almost_equal(tangent_vectors, true_tangent)
    
    # Plot

    print('test_orthogonal_vectors PASS')

def test_calculate_gradient(inward_distance = 1, parabolic=True):
    pc0_tangent_mag = np.asarray([0,0,0,0,0,0])
    pc1_tangent_mag = np.asarray([1,4,3,1,2, 0.46489556])
    pc2_tangent_mag = np.asarray([2,5,5,4,5, 0.60486809])

    g, xx, yy = wss_utils._calculate_gradient_with_values(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=parabolic)

    # Prepare to plot
    x = np.array([0,1,2]) # We have 3 points to evaluate
    x = x * inward_distance # Get the correct distance scaling
    y = np.stack((pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag), axis=1)
    
    # Plot the figures
    fig = plt.figure(1)
    main_title = "Parabolic" if parabolic else "Linear"
    main_title = "{} x={}".format(main_title, inward_distance)
    fig.suptitle(main_title)

    for i in range(0, len(y)):
        y_new = yy[i]
        g = np.gradient(y_new, xx)
        # print('wall_gradient', g[0])
        
        ax = fig.add_subplot(2,3,i+1)
        ax.plot(x,y[i],'o', xx, y_new)
        ax.title.set_text("Wall gradient {:.2f}".format(g[0]))
    
    plt.show()
    plt.clf()

if __name__ == "__main__":
    logging.getLogger("wss_utils").setLevel(logging.DEBUG)
    
    test_vector_magnitude()
    
    test_orthogonal_vectors()

    test_calculate_gradient(inward_distance=1, parabolic=True)
    test_calculate_gradient(inward_distance=0.7, parabolic=True)

    test_calculate_gradient(inward_distance=1, parabolic=False)
    test_calculate_gradient(inward_distance=0.7, parabolic=False)

    