# WorkApp2 TODO Items

## High Priority - Critical Architecture Violations

### **CRITICAL: Regex-Based Solutions Violating Core LLM Principles**

**Issue Discovered:** 2025-06-02  
**Reporter:** Code analysis via .clinerules violation audit  
**Status:** Multiple violations identified, refactoring required  

#### **Major Rule Violations in Core QA System:**

##### **1. `llm/pipeline/validation.py` - HEAVILY VIOLATES THE RULE**

This file has **15 regex operations** directly in the QA pipeline:

- `re.findall()` for extracting JSON blocks from LLM responses
- `re.sub()` for fixing malformed JSON (trailing commas, quotes, etc.)
- `re.search()` for pattern matching answer/confidence/sources
- This is core QA system functionality for processing LLM responses

##### **2. `llm/prompts/formatting_prompt.py` - VIOLATES THE RULE**

- `re.findall(r"^([A-Z][A-Za-z\s]+):$", raw_answer, re.MULTILINE)` - extracting section headers
- `re.search(r"Confidence: (\d+)%", raw_answer)` - finding confidence scores
- This is directly part of the answer formatting pipeline

##### **3. `llm/prompt_generator.py` - VIOLATES THE RULE**

- `re.split(r"(?<=[.!?]) +", context)` - splitting sentences
- `re.split(r"[.!?]+", context)` - splitting sentences
- This is part of prompt generation for the QA system

#### **The Irony**

The most problematic violation is in `llm/pipeline/validation.py` - this file is trying to "fix" LLM responses using regex patterns instead of letting the LLM handle its own output properly. This is exactly what the rule is meant to prevent!

#### **Acceptable Uses Found:**

- `utils/ui/text_processing.py` - UI display utilities
- `tests/test_*.py` - Testing frameworks
- `utils/common/validation_utils.py` - General validation
- `core/document_ingestion/` - Document preprocessing

#### **The Problem**

We've been using regex as a "band-aid" to fix LLM output issues instead of improving the LLM prompts and reasoning. This violates the core principle that the QA system should rely on LLM capabilities, not pattern matching.

---

## ✅ COMPLETED - LLM Formatting Pipeline Issue

### **FIXED: Formatting Prompt Echoing Instructions Instead of Following Them**

**Issue Discovered:** 2025-06-02  
**Reporter:** User via FFLBoss query investigation  
**Status:** ✅ **COMPLETED** - Fixed on 2025-06-02  
**Solution:** Moved conditional logic from LLM prompt to Python code  

#### **Problem Description**
The LLM formatting model is echoing back the raw prompt instructions instead of following them, causing malformed responses to be displayed to users.

#### **Root Cause Analysis**
Investigation of the FFLBoss query (session "3aec151e" in `logs/answer_pipeline_debug.json`) revealed:

1. ✅ **Extraction Phase**: Works correctly, returns proper JSON
2. ❌ **Formatting Phase**: Model echoes prompt instructions instead of formatting content

**Example malformed output:**
```
If the raw_answer reads exactly "Answer not found. Please contact a manager or fellow dispatcher.",
return it unchanged. Otherwise, format the answer as follows:

**Instructions:**
If a client mentions that they are part of FFL Boss in any form, they should submit a ticket either in the FFL Boss app or on the website(s) fflboss.com [URL: http://fflboss.com] or fflsoftwarepro.com [URL: http://fflsoftwarepro.com] for the quickest response.
Confidence Level: 100%
```

#### **Technical Root Cause**
Located in `llm/prompts/formatting_prompt.py` - the `generate_formatting_prompt()` function:

**Problematic prompt structure:**
```python
System: You are a formatting assistant for Karls Technology dispatchers.
        If the raw_answer reads exactly "Answer not found. Please contact a manager or fellow dispatcher.",
        return it unchanged. Otherwise, format the answer as follows:
```

**Issues:**
1. **Confusing conditional logic** - The LLM treats conditional instructions as content to include
2. **Verbose formatting rules** - Too many complex instructions confuse the model
3. **Ambiguous prompt boundaries** - Model doesn't distinguish between instructions and content to format

#### **Impact Assessment**
- **Severity**: High - Affects user experience with malformed responses
- **Scope**: Any query that goes through the formatting pipeline
- **Frequency**: Intermittent, but reproducible with certain content types
- **User Impact**: Confusion, unprofessional appearance

#### **Proposed Solution**

**Phase 1: Prompt Simplification**
1. **Restructure formatting prompt** to be clearer and more directive
2. **Remove conditional logic** from the main instruction text
3. **Simplify formatting rules** to reduce model confusion
4. **Use clearer instruction markers** to distinguish commands from content

**Example improved structure:**
```python
def generate_formatting_prompt(raw_answer: str) -> str:
    # Handle special case separately
    if raw_answer.strip() == "Answer not found. Please contact a manager or fellow dispatcher.":
        return raw_answer
    
    # Use simpler, clearer instructions
    return f"""Format the following text for display to Karls Technology dispatchers.

TEXT TO FORMAT:
{raw_answer}

INSTRUCTIONS:
- Use proper markdown formatting
- Format phone numbers as XXX-XXX-XXXX
- Use bullet points for lists
- Make section headers bold

FORMATTED TEXT:"""
```

**Phase 2: Testing & Validation**
1. **Create test cases** for known problematic content (FFLBoss, Surface Pro, etc.)
2. **A/B test** new prompt against current one
3. **Validate formatting quality** doesn't degrade
4. **Check edge cases** (empty content, special characters, etc.)

#### **Implementation Steps**

**Prerequisites:**
- [ ] Create comprehensive test suite for formatting pipeline
- [ ] Document current prompt behavior baseline
- [ ] Identify all content types that trigger the issue

**Phase 1: Core Fix**
- [ ] Refactor `generate_formatting_prompt()` in `llm/prompts/formatting_prompt.py`
- [ ] Update `check_formatting_quality()` validation logic if needed
- [ ] Test with FFLBoss query specifically
- [ ] Test with other known edge cases

**Phase 2: Comprehensive Testing**
- [ ] Run regression tests on existing queries
- [ ] Test with production data samples
- [ ] Validate no performance degradation
- [ ] Check formatting quality metrics

**Phase 3: Deployment**
- [ ] Deploy to test environment
- [ ] Monitor debug logs for formatting issues
- [ ] Gradual rollout if successful
- [ ] Update documentation

#### **Risk Assessment**
- **High Risk**: Changes affect all formatted responses
- **Mitigation**: Thorough testing, gradual rollout
- **Rollback Plan**: Keep current prompt as fallback
- **Monitoring**: Watch debug logs for formatting quality warnings

#### **Files Affected**
- `llm/prompts/formatting_prompt.py` (primary)
- `llm/pipeline/answer_pipeline.py` (testing integration)
- `logs/answer_pipeline_debug.json` (monitoring)

#### **Testing Checklist**
- [ ] FFLBoss queries return properly formatted content
- [ ] Surface Pro hardware queries format correctly
- [ ] Phone number queries maintain XXX-XXX-XXXX format
- [ ] Customer concern tickets format properly
- [ ] "Answer not found" responses remain unchanged
- [ ] Long procedural answers maintain readability
- [ ] Confidence scores display correctly

#### **Success Criteria**
1. **No prompt echo**: Formatting model follows instructions instead of echoing them
2. **Maintained quality**: Formatting quality check pass rate >= current baseline
3. **User experience**: No malformed responses visible to users
4. **Performance**: No degradation in response time or token usage

---

## Medium Priority Items

### **Future Enhancements**
- [ ] Implement more sophisticated prompt templates
- [ ] Add dynamic formatting based on content type
- [ ] Create specialized formatters for different domains (technical vs. procedural)

### **Monitoring Improvements**
- [ ] Add formatting quality metrics to debug logs
- [ ] Create alerts for prompt echoing detection
- [ ] Implement automated regression testing for formatting

---

**Last Updated**: 2025-06-02  
**Next Review**: After formatting fix implementation
