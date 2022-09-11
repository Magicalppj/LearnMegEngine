"""
2022-09-09 09:02:52 创建，用于规范分布式训练的工具箱
"""

import megengine.distributed as dist


def is_main_process():
    """
    判断分布式训练主线程
    :return:
    """
    return dist.get_rank() == 0


def reduce_value(value, average=True):
    """
    同步多GPU上的指定参数
    :param value:
    :param average:
    :return:
    """
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    dist.functional.all_reduce_sum(value)
    if average:
        value /= world_size

    return value
