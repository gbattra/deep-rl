# Greg Attra
# 03/26/22

'''
Executable for running linear function approximation on the four rooms problem
'''

from lib.function_approximation.four_rooms import FourRooms, four_rooms_arena


def main():
    arena = four_rooms_arena()
    env = FourRooms()


if __name__ == '__main__':
    main()
