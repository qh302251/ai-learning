import os
import tempfile
import unittest

# 让 Python 能找到 assistant 目录下的模块
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.local_tools import read_file, write_file, delete_file


class TestLocalTools(unittest.TestCase):
    def setUp(self):
        """每个测试前创建临时目录"""
        self.tmp_dir = tempfile.mkdtemp()

    def test_write_and_read(self):
        """写文件然后读回来,内容应该一致"""
        path = os.path.join(self.tmp_dir, "hello.txt")
        result = write_file(path, "你好，agent！")
        self.assertIn("写入成功", result)
        content = read_file(path)
        self.assertEqual(content, "你好，agent！")

    def test_write_file_create_parent_dir(self):
        """写入不存在的子目录，应该自动创建"""
        path = os.path.join(self.tmp_dir, "sub", "nested", "file.txt")
        result = write_file(path, "深层目录")
        self.assertIn("写入成功", result)
        self.assertTrue(os.path.exists(path))    

    def test_delete_file(self):
        """写完文件再删除，确认文件不存在"""
        path = os.path.join(self.tmp_dir, "to_delete.txt")
        write_file(path, "待删除")
        delete_file(path)
        self.assertFalse(os.path.exists(path))

    def test_delete_file_return_message(self):
        """写完文件再删除，验证删除返回的信息"""
        path = os.path.join(self.tmp_dir, "tmp.txt")
        write_file(path, "内容")
        result=delete_file(path)
        self.assertIn("已删除", result)    

    def test_delete_not_exist_file(self):
        """删除不存在的文件"""
        path = os.path.join(self.tmp_dir, "不存在.txt")
        result=delete_file(path)
        self.assertIn("不存在", result) 


    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir,ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
