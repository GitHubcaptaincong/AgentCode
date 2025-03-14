import warnings
warnings.filterwarnings("ignore")

from langchain.tools import StructuredTool
from .FileQATool import ask_docment
from .ExcelTool import get_first_n_rows
from .FileTool import list_files_in_directory
from .FinishTool import finish

document_qa_tool = StructuredTool.from_function(
    func=ask_docment,
    name="AskDocument",
    description="根据一个Word或PDF文档的内容，回答一个问题。考虑上下文信息，确保问题对相关概念的定义表述完整。",
)

excel_inspection_tool = StructuredTool.from_function(
    func=get_first_n_rows,
    name="InspectExcel",
    description="探查表格文件的内容和结构，展示它的列名和前n行，如果你找不到数据结果，可以传入更大的参数n探查更多的数据",
)

directory_inspection_tool = StructuredTool.from_function(
    func=list_files_in_directory,
    name="ListDirectory",
    description="探查文件夹的内容和结构，展示它的文件名和文件夹名",
)

finish_placeholder = StructuredTool.from_function(
    func=finish,
    name="FINISH",
    description="当任务完成时，总结回答，并打印最终的答案"
)
