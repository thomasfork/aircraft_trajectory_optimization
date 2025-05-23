'''
shader programs for OpenGL Figure
'''
import numpy as np

vtype = [('a_position', np.float32, 3),
         ('a_normal',   np.float32, 3),
         ('a_color',    np.float32, 4)]

DEFAULT_VERTEX = """
    #version 330 core

    layout (location = 0) in vec3 a_position;      // Vertex position
    layout (location = 1) in vec3 a_normal;        // Vertex normal
    layout (location = 2) in vec4 a_color;         // Vertex color

    out vec3 v_pos;
    out vec4 v_color;
    out vec3 v_normal;

    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };
    uniform mat4   u_model; // Model matrix

    void main()
    {
        v_pos    = vec3(u_model * vec4(a_position, 1.0));
        v_color  = a_color;
        v_normal = vec3(u_model * vec4(a_normal, 0.0));

        gl_Position = u_projection * u_view * vec4(v_pos, 1.0);
    }
    """

DEFAULT_FRAGMENT = """
    #version 330 core

    in vec3 v_pos;
    in vec4 v_color;
    in vec3 v_normal;

    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };

    const vec3 light_dir = vec3(0,0,1);
    const vec3 light_color = vec3(1,1,1);
    
    out vec4 FragColor;

    void main()
    {
        // ambient
        vec3 ambient = 0.1 * light_color;

        // diffuse
        vec3 normal = normalize(v_normal);
        vec3 diffuse = max(dot(normal, light_dir), 0.0) * light_color;

        vec3 view_pos = - transpose(mat3(u_view)) * vec3(u_view[3][0], u_view[3][1], u_view[3][2]);

        // specular
        vec3 view_dir = normalize(view_pos - v_pos);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
        vec3 specular = 0.5 * spec * light_color;

        vec3 result = (ambient + diffuse + specular) * vec3(v_color);

        FragColor = vec4(result, v_color.w);
    }
    """

SIMPLE_FRAGMENT = """
    #version 330 core
    in vec4 v_color;
    
    out vec4 FragColor;

    void main()
    {
        FragColor = v_color;

    }
    """

INSTANCED_VERTEX = """
    #version 330 core

    layout (location = 0) in vec3 a_position;      // Vertex position
    layout (location = 1) in vec3 a_normal;        // Vertex normal
    layout (location = 2) in vec4 a_color;         // Vertex color
    layout (location = 3) in mat4 a_instance_matrix;

    out vec3 v_pos;
    out vec4 v_color;
    out vec3 v_normal;

    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };
    uniform mat4   u_model; // Model matrix

    void main()
    {
        v_pos    = vec3(u_model * a_instance_matrix * vec4(a_position, 1.0));
        v_color  = a_color;
        v_normal = vec3(u_model * a_instance_matrix * vec4(a_normal, 0.0));

        gl_Position = u_projection * u_view * vec4(v_pos, 1.0);
    }
    """

SKYBOX_VERTEX = """
    #version 330 core
    in vec3 a_position;

    out vec3 tex_coords;

    uniform mat4 u_projection;
    uniform mat4 u_view;
    uniform mat4 u_model;

    void main()
    {
        gl_Position = vec4(a_position, 1.0);
        tex_coords = mat3(u_model) * mat3(u_view) * vec3(u_projection * vec4(a_position, 1.0));
    }
    """

SKYBOX_FRAGMENT = """
    #version 330 core
    in vec3 tex_coords;
    uniform samplerCube skybox;
    
    out vec4 FragColor;
    
    void main()
    {
        FragColor = texture(skybox, tex_coords);
    }
    """

POINTCLOUD_VERTEX = """
    #version 330 core

    layout (location = 0) in vec3 a_position;      // Vertex position

    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };
    uniform mat4   u_model;         // Model matrix

    void main()
    {
        gl_Position = u_projection * u_view * u_view * vec4(a_position, 1.0);
    }

    """

POINTCLOUD_FRAGMENT = """
    #version 330 core
    
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(1,0,0,1);
    }
    """

DEPTH_CAM_VERTEX = """
    #version 330 core
    layout (location = 0) in vec2 a_position;

    uniform sampler2D depth;
    
    out vec2 v_texcoord;
    out float interp_depth;
    
    layout (std140) uniform Matrices
    {
        mat4 u_projection;  // Projection matrix
        mat4 u_view;        // View matrix
    };
    uniform mat4   u_model; // Model matrix

    uniform float fx;
    uniform float fy;
    uniform float ppx;
    uniform float ppy;
    uniform float width;
    uniform float height;
    uniform float max_depth;

    void main()
    {
        float x = (a_position[0] * width  - ppx) / fx;
        float y = (a_position[1] * height - ppy) / fy;
        interp_depth = max_depth * texture(depth, a_position).r;

        vec3 v_pos    = vec3(u_model * vec4(x*interp_depth, y*interp_depth, interp_depth, 1.0));
        //vec3 v_pos    = vec3(u_model * vec4(x, y, 0, 1.0));
        v_texcoord = a_position;
        
        gl_Position = u_projection * u_view * vec4(v_pos, 1.0);
       
    }


    """

DEPTH_CAM_FRAGMENT = """
    #version 330 core
    uniform sampler2D color;

    in vec2 v_texcoord;
    in float interp_depth;
    
    out vec4 FragColor;

    void main()
    {
        FragColor = texture(color, v_texcoord);
        if (interp_depth == 0){
            FragColor.a = 0.0;
        }
    }
    """
