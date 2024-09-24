# swe_actions.py

import os
import logging
import subprocess
from chat_with_ollama import ChatGPT
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from swe_actions.file_operations import CreateFileAction, EditFileAction, AddFunctionAction, HandleCodeInsertionAction
from swe_actions.git_operations import CreateBranchAction, MergeBranchAction, PushChangesAction, PullChangesAction, ResolveMergeConflictsAction
from swe_actions.testing_quality import WriteTestsAction, RunTestsAction, AnalyzeTestCoverageAction, UpdateDependenciesAction
from swe_actions.code_modification import RefactorCodeAction, FormatCodeAction, AnalyzeCodeQualityAction
from swe_actions.dependency_management import ManageDependenciesAction
from swe_actions.deployment import DockerizeApplicationAction, DeployApplicationAction, MonitorLogsAction, CommitChangesAction
from swe_actions.issue_management import CreateIssueAction, AssignTaskAction, RevertChangesAction
from swe_actions.documentation import GenerateDocsAction
from swe_actions.code_execution import SafeExecuteCodeAction
from swe_actions.text_generation import GenerateContentAction, CodeReviewAction
@dataclass
class Context:
    response: str
    project_directory: str
    entities: Dict[str, Any]
    # Add other shared resources as needed


class Action(ABC):
    @abstractmethod
    def execute(self, context: Dict):
        pass

    @staticmethod
    def extract_file_name(text):
        import re
        match = re.search(r'file named (\w+\.\w+)', text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_function_name(text):
        import re
        match = re.search(r'function called (\w+)', text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_code_snippet(text):
        import re
        match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_file_path(text):
        import re
        match = re.search(r'in file ([\w./]+)', text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_edit_instructions(self, text):
        """
        Extracts edit instructions from the text.
        """
        # For simplicity, we'll assume the entire text is the edit instruction
        return text

    
class ActionRegistry:
    def __init__(self):
        self._actions = {}
    
    def register_action(self, name: str, action: Action):
        self._actions[name] = action
    
    def get_action(self, name: str) -> Action:
        return self._actions.get(name)
    
    def list_actions(self) -> List[str]:
        return list(self._actions.keys())

def register_actions():
    registry = ActionRegistry()
    registry.register_action("create_file", CreateFileAction())
    registry.register_action("generate_content", GenerateContentAction())
    registry.register_action("add_function", AddFunctionAction())
    registry.register_action("edit_file", EditFileAction())
    registry.register_action("write_tests", WriteTestsAction())
    registry.register_action("handle_code_insertion", HandleCodeInsertionAction())
    registry.register_action("run_tests", RunTestsAction())
    registry.register_action("generate_docs", GenerateDocsAction())
    registry.register_action("safe_execute_code", SafeExecuteCodeAction())
    registry.register_action("commit_changes", CommitChangesAction())
    registry.register_action("create_branch", CreateBranchAction())
    registry.register_action("merge_branch", MergeBranchAction())
    registry.register_action("push_changes", PushChangesAction())
    registry.register_action("pull_changes", PullChangesAction())
    registry.register_action("resolve_merge_conflicts", ResolveMergeConflictsAction())
    registry.register_action("code_review", CodeReviewAction())
    registry.register_action("refactor_code", RefactorCodeAction())
    registry.register_action("analyze_code_quality", AnalyzeCodeQualityAction())
    registry.register_action("format_code", FormatCodeAction())
    registry.register_action("manage_dependencies", ManageDependenciesAction())
    registry.register_action("dockerize_application", DockerizeApplicationAction())
    registry.register_action("deploy_application", DeployApplicationAction())
    registry.register_action("monitor_logs", MonitorLogsAction())
    registry.register_action("create_issue", CreateIssueAction())
    registry.register_action("assign_task", AssignTaskAction())
    registry.register_action("revert_changes", RevertChangesAction())
    registry.register_action("analyze_test_coverage", AnalyzeTestCoverageAction())
    registry.register_action("update_dependencies", UpdateDependenciesAction())
    return registry



