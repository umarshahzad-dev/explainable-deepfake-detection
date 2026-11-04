import unittest
from PIL import Image
from tests.support import module
from tests import test_extraction_retries as support


class RetryMemoryTests(unittest.TestCase):
    def test_video_reader_continues_absolute_position_between_segments(self):
        import numpy as np
        from unittest.mock import patch
        import cv2
        class Capture:
            index=-1
            def isOpened(self): return True
            def get(self,key): return 5 if key==cv2.CAP_PROP_FRAME_COUNT else 1
            def grab(self): self.index+=1;return self.index<5
            def retrieve(self): return True,np.full((2,2,3),self.index,dtype=np.uint8)
            def release(self): pass
        with patch("cv2.VideoCapture",return_value=Capture()):
            with module(self,"extraction").VideoReader("generated") as reader:
                first=list(reader.frames([1]));second=list(reader.frames([3]))
        self.assertEqual(first[0][0],1)
        self.assertEqual(second[0][0],3)
        self.assertEqual(second[0][1].getpixel((0,0)),(3,3,3))
    def frames(self,size):
        class Reader:
            def frames(self,targets):
                for index in sorted(targets): yield index,Image.new("RGB",size)
        return Reader()

    def test_many_large_frames_trigger_budget_exhaustion(self):
        stats={}
        result=list(module(self,"extraction").segment_frames(self.frames((512,512)),[[0,1,2,3,4]],
                    limits={"retry_max_frames_in_memory":8,"retry_max_pixels_per_frame":4000000,"retry_frame_memory_budget_mb":1.5},stats=stats))
        self.assertEqual(result,[])
        self.assertEqual(stats["retry_budget_exhaustions"],1)

    def test_small_frames_complete(self):
        stats={}
        result=list(module(self,"extraction").segment_frames(self.frames((50,50)),[[0,1,2]],stats=stats))
        self.assertEqual(len(result),3)
        self.assertEqual(stats["retry_budget_exhaustions"],0)

    def test_peak_memory_is_bounded(self):
        stats={}
        list(module(self,"extraction").segment_frames(self.frames((512,512)),[[0,1,2,3,4]],
             limits={"retry_max_frames_in_memory":8,"retry_max_pixels_per_frame":4000000,"retry_frame_memory_budget_mb":1.5},stats=stats))
        self.assertLessEqual(stats["peak_retry_memory_mb"],1.5)
        self.assertGreater(stats["peak_retry_memory_mb"],0)

    def test_extraction_reports_exhaustion(self):
        ws,cfg,base=support.RetryTests.setup_video(self)
        cfg["extraction"].update(retry_max_frames_in_memory=8,retry_max_pixels_per_frame=100,retry_frame_memory_budget_mb=512)
        result=module(self,"extraction").extract("fixture",cfg,ws,detector=object(),reader_factory=base)
        self.assertEqual(result["retry_budget_exhaustions"],1)
        self.assertEqual(result["peak_retry_memory_mb"],0)
