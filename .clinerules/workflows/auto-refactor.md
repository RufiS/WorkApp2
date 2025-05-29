# Intelligent Code Refactoring Workflow

This workflow automatically refactors large Python files (>700 lines) by intelligently analyzing code structure and extracting logical components. It adapts to ANY file type and discovers optimal organization patterns.

---

## Phase 1: Smart File Discovery

Let me scan your codebase to find files that need refactoring:

<search_files>
<path>.</path>
<regex>.*</regex>
<file_pattern>*.py</file_pattern>
</search_files>

Now I'll analyze which files are over 700 lines and need refactoring. Let me check the line counts of the largest files:

Based on my analysis, I found these files that need refactoring:

<ask_followup_question>
<question>I found several large Python files. Which would you like me to refactor?</question>
<options>["Show me all files over 700 lines first", "Refactor the largest file", "Refactor all large files", "Let me specify a file"]</options>
</ask_followup_question>

---

## Phase 2: Dynamic Content Analysis

I'll read and analyze the structure of your selected file to understand what can be extracted:

<read_file>
<path>SELECTED_FILE_PATH</path>
</read_file>

**Intelligent Analysis Complete!**

Based on my analysis of this file, I can identify the following extractable components:

- **[Discovery will show actual components found]**
- **[Not predefined categories, but what's actually there]**

Let me also examine your existing project structure to see where extractions should go:

<search_files>
<path>.</path>
<regex>^[^/]+/$</regex>
</search_files>

**Project Structure Analysis:**
I can see your existing directories and will suggest extractions that fit your current organization patterns.

<ask_followup_question>
<question>I've analyzed the file structure. Here's what I found that can be extracted:

[DYNAMIC RESULTS BASED ON ACTUAL ANALYSIS]

How would you like to proceed with the extractions?</question>
<options>["Show me detailed extraction plan", "Extract the largest components first", "Use my existing directory structure", "Let me customize the plan"]</options>
</ask_followup_question>
---

## Phase 3: Safety Backup Creation

Before making any changes, I'll create a backup of the original file:

<execute_command>
<command>cp SELECTED_FILE_PATH SELECTED_FILE_PATH.backup</command>
<requires_approval>false</requires_approval>
</execute_command>

Backup created successfully! You can restore with: `cp SELECTED_FILE_PATH.backup SELECTED_FILE_PATH`

---

## Phase 4: Adaptive Directory Planning

Let me check your existing directory structure to plan optimal extractions:

<search_files>
<path>.</path>
<regex>^[^/]+/$</regex>
</search_files>

Based on your existing structure and the components I found, I recommend:

**[DYNAMIC RECOMMENDATIONS BASED ON ACTUAL ANALYSIS]**

<ask_followup_question>
<question>I can extract components to fit your existing structure. Should I proceed with these recommendations or would you like to modify the plan?</question>
<options>["Proceed with recommended structure", "Use different directories", "Create new directories", "Show me more details"]</options>
</ask_followup_question>

---

## Phase 5: Intelligent Component Extraction

I'll now extract the components, handling large files safely to avoid truncation issues.

### Step 5.1: Extract First Component

Based on my analysis, I'll start with the largest/most logical component to extract:

**[COMPONENT EXTRACTION WILL BE DYNAMIC BASED ON ACTUAL ANALYSIS]**

<write_to_file>
<path>DYNAMIC_TARGET_PATH</path>
<content>
"""
[ACTUAL EXTRACTED CODE CONTENT]

Extracted from SELECTED_FILE_PATH
"""

[REAL CODE CONTENT BASED ON ANALYSIS]
---

## Phase 6: Import Updates and Validation

I'll search for any other files that import from the refactored file and update them:

<search_files>
<path>.</path>
<regex>from SELECTED_FILE_NAME import|import SELECTED_FILE_NAME</regex>
<file_pattern>*.py</file_pattern>
</search_files>

For each file that needs import updates, I'll fix the imports to point to the new locations:

<replace_in_file>
<path>AFFECTED_FILE_PATH</path>
<diff>
{
  "replacements": [
    {
      "old_str": "from SELECTED_FILE_NAME import COMPONENT_NAME",
      "new_str": "from DYNAMIC_MODULE import COMPONENT_NAME"
    }
  ]
}
</diff>
</replace_in_file>

---

## Phase 7: Final Validation

Let me verify the refactored code works correctly:

<ask_followup_question>
<question>I've completed the refactoring. Would you like me to validate the changes?</question>
<options>["Yes, run validation tests", "No, I'll test manually", "Show me a summary first", "Make more extractions"]</options>
</ask_followup_question>

If validation is requested, I'll check syntax and imports:

<execute_command>
<command>python -c "import SELECTED_FILE_NAME; print('✅ Main file imports successfully')"</command>
<requires_approval>false</requires_approval>
</execute_command>

<execute_command>
<command>python -c "import EXTRACTED_MODULE; print('✅ Extracted module imports successfully')"</command>
<requires_approval>false</requires_approval>
</execute_command>

---

## Phase 8: Summary Report

**Refactoring Complete!** ✅

**What was accomplished:**
- **Original file size:** [ORIGINAL_LINE_COUNT] lines
- **New file size:** [NEW_LINE_COUNT] lines
- **Reduction:** [REDUCTION_PERCENTAGE]% smaller
- **Components extracted:** [NUMBER_OF_EXTRACTIONS]
- **Files created:** [LIST_OF_NEW_FILES]
- **Files updated:** [LIST_OF_UPDATED_FILES]

**Key Features:**
- ✅ **Content-driven analysis** - Discovered actual components, not assumptions
- ✅ **Adaptive directory structure** - Used existing patterns where possible
- ✅ **Safe extraction** - Created backups and validated changes
- ✅ **Smart import updates** - Fixed all references automatically
- ✅ **Interactive decisions** - Asked for approval at key steps

**Files created/modified:**
- `DYNAMIC_TARGET_PATH` - [DESCRIPTION]
- `SELECTED_FILE_PATH` - Refactored original file
- `SELECTED_FILE_PATH.backup` - Safety backup

**To rollback if needed:**
```bash
cp SELECTED_FILE_PATH.backup SELECTED_FILE_PATH
rm DYNAMIC_TARGET_PATH
```

**Next steps:**
1. Test the refactored application thoroughly
2. Run any existing test suites
3. Consider refactoring other large files using `/auto-refactor.md`
4. Remove backup files once satisfied with results

This workflow successfully demonstrates intelligent, content-driven refactoring that adapts to any file type and project structure!
