import aiohttp                          # For asynchronously making HTTP requests
import asyncio
import concurrent.futures               # Allows creating new processes
from multiprocessing import cpu_count   # Returns our number of CPU cores
                                        # Helps divide up our requests evenly across our CPU cores
from math import floor
import numpy as np
import os
import cv2
import requests
import timeit

cam_addr_list = ["http://192.168.88.133", "http://192.168.88.246",
                "http://192.168.88.136", "http://192.168.88.228",
                "http://192.168.88.123", "http://192.168.88.124",
                "http://192.168.88.143", "http://192.168.88.254",
                "http://192.168.88.13", "http://192.168.88.214",
                "http://192.168.88.33", "http://192.168.88.24",
                "http://192.168.88.3", "http://192.168.88.22",
                "http://192.168.88.23", "http://192.168.88.2",
                "http://192.168.88.43", "http://192.168.88.16",
                "http://192.168.88.11", "http://192.168.88.26"]

# cam_addr_list = ["http://192.168.88.133", "http://192.168.88.246"]

img_array = np.empty(len(cam_addr_list), dtype=object)

timeout = aiohttp.ClientTimeout(total=1.5)

async def get_response(index):
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(cam_addr_list[index]+"/capture") as response:
                # response.raise_for_status()
                print(f"Response status ({cam_addr_list[index]}): {response.status}")
                image = np.asarray(bytearray(await response.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            image = np.zeros((1600, 1200, 3), dtype="uint8")
        
        except aiohttp.ClientConnectorError as e:
            print('Connection Error', str(e))
            image = np.zeros((1600, 1200, 3), dtype="uint8")
            
        except Exception as err:
            print(f"An error ocurred: {err}")
            image = np.zeros((1600, 1200, 3), dtype="uint8")
        
        img_array[index] = image

async def request(index):
    if index == 0:
        await asyncio.gather(
            get_response(0),
            get_response(1),
            get_response(2),
            get_response(3),
            get_response(4),
        )
    elif index == 1:
        await asyncio.gather(
            get_response(5),
            get_response(6),
            get_response(7),
            get_response(8),
            get_response(9),
        )
    elif index == 2:
        await asyncio.gather(
            get_response(10),
            get_response(11),
            get_response(12),
            get_response(13),
            get_response(14),
        )
    elif index == 3:
        await asyncio.gather(
            get_response(15),
            get_response(16),
            get_response(17),
            get_response(18),
            get_response(19),
        )


def start_request(index):
    asyncio.run(request(index))

# def main():
#     for i in range(len(cam_addr_list)):
#         start_request(i)

def main():
    NUM_CORES = cpu_count() # Our number of CPU cores (including logical cores)
    NUM_URL = len(cam_addr_list)

    futures = [] # To store our futures

    with concurrent.futures.ProcessPoolExecutor(NUM_CORES) as executor:
        for i in range(NUM_CORES):
            new_future = executor.submit(
                start_request, # Function to perform
                # v Arguments v
                i
            )
            futures.append(new_future)

    concurrent.futures.wait(futures)

if __name__ == "__main__":
    print(timeit.timeit(main, number=10))
    # main()
    # print(img_array)
    # for i,img in enumerate(img_array):
    #     cv2.imshow(str(i), img)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


