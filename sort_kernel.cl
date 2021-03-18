__kernel void bitonic_sort_kernel_default(__global int * data, uint stage, uint passOfStage, uint direction)
{
    //  --------------------------------------------------------------------
    //  | This is default kernel from book "OpenCL Programmimg by Example".|
    //  | This kernel don't use local memory, because of this this kernel  |
    //  | is a bit slow as there is a frequent access to the global memory.|
    //  --------------------------------------------------------------------

    uint id = get_global_id(0);
    
    uint pairDistance = 1 << (stage - passOfStage);

    uint left_id = (id % pairDistance) + (id / pairDistance) * 2 * pairDistance;
    uint right_id = left_id + pairDistance;
    
    int left_elem = data[left_id];
    int right_elem = data[right_id];
    
    if((id / (1 << stage)) % 2 == 1)
        direction = 1 - direction;

    int greater = (left_elem > right_elem) ? left_elem : right_elem;
    int lesser = (left_elem > right_elem) ? right_elem : left_elem;

    data[left_id] = direction ? lesser : greater;
    data[right_id] = direction ? greater : lesser;
}

__kernel void bitonic_sort_kernel_local(__global int* data, __local int* local_data, uint numStages, uint direction) {

    //  -------------------------------------------------
    //  | Default sort from book, but with local memory. |
    //  -------------------------------------------------

    uint local_id = get_local_id(0);
    uint work_group_size = get_local_size(0);
    uint offset = get_group_id(0) * work_group_size;

    local_data[local_id] = data[offset * 2 + local_id];
    local_data[local_id + work_group_size] = data[offset * 2 + local_id + work_group_size];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint stage = 0; stage < numStages; ++stage) {
        for(uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            uint dir = direction;
            uint pairDistance = 1 << (stage - passOfStage);

            uint left_id = (local_id % pairDistance) + (local_id / pairDistance) * 2 * pairDistance;
            uint right_id = left_id + pairDistance;

            int left_elem = local_data[left_id];
            int right_elem = local_data[right_id];

            if((local_id / (1 << stage)) % 2 == 1)
                dir = 1 - dir;

            int greater = (left_elem > right_elem) ? left_elem : right_elem;
            int lesser = (left_elem > right_elem) ? right_elem : left_elem;

            local_data[left_id] = dir ? lesser : greater;
            local_data[right_id] = dir ? greater : lesser;

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    data[offset * 2 + local_id] = local_data[local_id];
    data[offset * 2 + local_id + work_group_size] = local_data[local_id + work_group_size];
}
