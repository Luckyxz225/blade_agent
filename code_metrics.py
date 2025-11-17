#!/usr/bin/env python3
"""
代码度量工具 (Code Metrics Tool)
=================================

用途：自动生成项目代码的各种度量指标
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import ast


class CodeMetricsAnalyzer:
    """代码度量分析器"""
    
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.metrics = {
            "summary": {},
            "files": [],
            "dependencies": defaultdict(int),
            "functions": [],
            "classes": []
        }
    
    def analyze(self):
        """执行完整分析"""
        print("🔍 开始代码分析...")
        
        self._analyze_python_files()
        self._calculate_summary()
        self._generate_report()
        
        print("✅ 分析完成！")
        return self.metrics
    
    def _analyze_python_files(self):
        """分析所有 Python 文件"""
        python_files = list(self.root_dir.rglob("*.py"))
        
        for file_path in python_files:
            # 跳过 .git 目录
            if ".git" in str(file_path):
                continue
            
            try:
                self._analyze_file(file_path)
            except Exception as e:
                print(f"⚠️  分析文件失败 {file_path}: {e}")
    
    def _analyze_file(self, file_path):
        """分析单个文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基础统计
        lines = content.split('\n')
        total_lines = len(lines)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        blank_lines = total_lines - code_lines - comment_lines
        
        file_info = {
            "path": str(file_path.relative_to(self.root_dir)),
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
            "functions": 0,
            "classes": 0
        }
        
        # AST 分析
        try:
            tree = ast.parse(content)
            
            # 统计函数和类
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    file_info["functions"] += 1
                    self.metrics["functions"].append({
                        "name": node.name,
                        "file": file_info["path"],
                        "line": node.lineno,
                        "args": len(node.args.args)
                    })
                elif isinstance(node, ast.ClassDef):
                    file_info["classes"] += 1
                    self.metrics["classes"].append({
                        "name": node.name,
                        "file": file_info["path"],
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.metrics["dependencies"][alias.name] += 1
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.metrics["dependencies"][node.module] += 1
        
        except SyntaxError:
            pass  # 语法错误的文件跳过
        
        self.metrics["files"].append(file_info)
    
    def _calculate_summary(self):
        """计算汇总统计"""
        files = self.metrics["files"]
        
        self.metrics["summary"] = {
            "total_files": len(files),
            "total_lines": sum(f["total_lines"] for f in files),
            "total_code_lines": sum(f["code_lines"] for f in files),
            "total_comment_lines": sum(f["comment_lines"] for f in files),
            "total_blank_lines": sum(f["blank_lines"] for f in files),
            "total_functions": len(self.metrics["functions"]),
            "total_classes": len(self.metrics["classes"]),
            "avg_lines_per_file": sum(f["total_lines"] for f in files) / len(files) if files else 0,
            "comment_ratio": sum(f["comment_lines"] for f in files) / sum(f["total_lines"] for f in files) if files else 0,
            "analyzed_at": datetime.now().isoformat()
        }
    
    def _generate_report(self):
        """生成报告"""
        summary = self.metrics["summary"]
        
        print("\n" + "="*60)
        print("📊 代码度量报告 (Code Metrics Report)")
        print("="*60)
        
        print(f"\n📁 文件统计:")
        print(f"  - 总文件数: {summary['total_files']}")
        print(f"  - 总代码行数: {summary['total_code_lines']}")
        print(f"  - 总注释行数: {summary['total_comment_lines']}")
        print(f"  - 总空行数: {summary['total_blank_lines']}")
        print(f"  - 平均每文件行数: {summary['avg_lines_per_file']:.1f}")
        print(f"  - 注释率: {summary['comment_ratio']*100:.1f}%")
        
        print(f"\n🔧 结构统计:")
        print(f"  - 函数总数: {summary['total_functions']}")
        print(f"  - 类总数: {summary['total_classes']}")
        
        print(f"\n📦 最常用依赖 (Top 10):")
        top_deps = sorted(self.metrics["dependencies"].items(), 
                         key=lambda x: x[1], reverse=True)[:10]
        for dep, count in top_deps:
            print(f"  - {dep}: {count} 次")
        
        # 找出最大的文件
        if self.metrics["files"]:
            largest_files = sorted(self.metrics["files"], 
                                  key=lambda x: x["total_lines"], 
                                  reverse=True)[:5]
            print(f"\n📄 最大的 5 个文件:")
            for f in largest_files:
                print(f"  - {f['path']}: {f['total_lines']} 行")
    
    def export_json(self, output_path="code_metrics.json"):
        """导出为 JSON"""
        output_file = self.root_dir / output_path
        
        # 转换 defaultdict 为普通 dict
        export_data = {
            "summary": self.metrics["summary"],
            "files": self.metrics["files"],
            "dependencies": dict(self.metrics["dependencies"]),
            "top_functions": sorted(self.metrics["functions"], 
                                   key=lambda x: x.get("args", 0), 
                                   reverse=True)[:20],
            "classes": self.metrics["classes"]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 度量数据已导出到: {output_file}")


def main():
    """主函数"""
    analyzer = CodeMetricsAnalyzer()
    analyzer.analyze()
    analyzer.export_json()


if __name__ == "__main__":
    main()
