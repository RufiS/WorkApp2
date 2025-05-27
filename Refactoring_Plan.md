Refactoring Plan for WorkApp2 Codebase
1. Code Deduplication (Eliminate Redundant Logic)

Merge LLM Service modules ( llm_service.py & llm_service_enhanced.py ) – These two
files implement the same LLMService with overlapping code. For example, both define
validate_json_output (duplicated JSON validation logic).
 Why: Removing one copy prevents divergence and eases maintenance. 
 How: Create a single llm_service.py that incorporates all enhancements from the "enhanced" version (e.g. context truncation before prompts, check_formatting_quality usage, stricter input validation) into the base class. 
 Update any imports (e.g. in workapp3.py ) to use the unified module. 
 Delete the old llm_service_enhanced.py to avoid confusion.
Unify Document Processing & File Handling – Consolidate the document ingestion code that currently
exists in multiple places. The class DocumentProcessor in document_processor_unified.py
duplicates functionality from file_processing.py and caching.py (e.g. both define
ChunkCacheEntry , chunking and hashing functions). Why: Keeping one implementation of file
loading, chunk splitting, and caching simplifies debugging and ensures consistent behavior. How:
Use the DocumentProcessor class as the single source of truth for document loading and
chunking. Merge any unique logic from file_processing.py (such as file type handling or chunk
post-processing) into DocumentProcessor methods ( _get_file_loader ,
load_and_chunk_document , _handle_small_chunks , etc.). Likewise, use the
ChunkCacheEntry defined in DocumentProcessor and remove the duplicate definition in
caching.py . After integrating, remove the now-redundant file_processing.py and utils/
caching.py modules. Ensure all code (e.g. Streamlit file uploads in workapp3.py ) calls the
unified DocumentProcessor instead of any standalone functions.
Consolidate Retrieval Logic (Hybrid Search) – There are two retrieval systems:
EnhancedRetrieval (in enhanced_retrieval.py ) and UnifiedRetrievalSystem (in
unified_retrieval_system.py ). They overlap in purpose – both perform combined vector and
keyword search – but maintain separate implementations (e.g. both handle BM25 vs. FAISS searches
and re-ranking). Why: Maintaining two versions of retrieval code risks inconsistency (bugs fixed in
one not in the other) and adds complexity. How: Merge these into a single retrieval module/class.
Prefer the UnifiedRetrievalSystem class as the base (since the Streamlit app already uses it)
and incorporate any features from EnhancedRetrieval not present in the unified version. For
example, if EnhancedRetrieval.hybrid_search or mmr_reranking logic is superior,
integrate that into UnifiedRetrievalSystem.retrieve or a new helper method
_rerank_results . Likewise, include any “keyword fallback” search functionality so that the
unified class covers all scenarios. Once unified, update references in code to use the one class and
remove the outdated module. This results in one clear Retrieval Engine for the app.
•
•
•
1
Merge Prompt Generators ( formatting_prompt.py & formatting_prompt_enhanced.py ) –
Both files define generate_formatting_prompt with minor differences, and the enhanced version
adds check_formatting_quality . Why: We should have one consistent prompt format for
answer formatting to avoid confusion. How: Unify into a single formatting_prompt.py module.
Take the base generate_formatting_prompt and extend it with the improvements from the
enhanced version (for instance, regex-based post-processing or better handling of bullet points if
present). If check_formatting_quality is a useful utility for evaluating the formatted answer,
include it in this module (or integrate its logic directly into the formatting workflow if appropriate).
Update LLMService to import from the unified prompt module. Remove
formatting_prompt_enhanced.py after confirming the new unified prompt works as expected.
Use One Set of Error-Handling Decorators – The utils/error_handling/ package has duplicate
decorator logic: decorators.py (original) vs enhanced_decorators.py (improved). For
example, with_retry vs with_advanced_retry , and differing error tracking wrappers. Why:
Standardizing on one robust set of decorators simplifies error handling and avoids confusion about
which to use. How: Adopt the enhanced decorators as the default (they appear to provide advanced
retry logic, timing, and error tracking). Refactor code to use these consistently: e.g. replace
@with_retry with @with_advanced_retry and @with_error_handling with
@with_error_tracking where applicable. If the older decorators have any functionality not in
the new ones (such as specific exception filtering or recovery hooks), extend the enhanced versions
to cover those needs. For instance, in DocumentProcessor._update_metadata_with_hash ,
which currently uses @with_retry to catch file I/O issues, use with_advanced_retry and
specify the exception types in its parameters (modify the decorator if needed to accept a tuple of
exception classes). Once all usages are migrated, remove references to decorators.py and
consider deleting it. This ensures all retry/error-handling logic is defined in one module
( enhanced_decorators.py ) with a consistent implementation.
Unify UI Helper Functions – The UI layer has duplicated component functions: ui/components.py
vs ui/enhanced_components.py . For example, components.py defines generic display
functions ( display_answer , display_confidence_meter , etc.), while
enhanced_components.py provides more advanced or additional displays
( display_enhanced_answer , display_search_results , etc.). Why: Having two sets of
Streamlit UI functions complicates the UI code and may lead to inconsistent user experience if some
features use old components. How: Merge the relevant UI functions into a single module (e.g. keep
components.py and add enhanced features to it). Identify any overlapping functionality:
If an “enhanced” function is essentially a better version of an original (e.g.
display_enhanced_answer vs display_answer ), replace or upgrade the original function
accordingly (preserve the interface but incorporate richer formatting or additional info like sources
or highlighted text).
For new functionality in enhanced (like display_error_message or display_system_status ),
migrate those into the unified components module as well. After merging, update the Streamlit app
( workapp3.py ) to import from the unified components module only. Remove
enhanced_components.py to avoid confusion. The result is a single set of UI helper functions
with all features, making the interface code easier to follow.
•
•
•
•
•
2
2. Improve Modular Organization & Decoupling
Structure Code by Feature – Reorganize modules into logical groups rather than a flat utils . Why:
A clearer project structure helps new contributors find relevant code quickly and encourages
separation of concerns. How: Consider organizing into packages like:
core/ (or services/ ): for non-UI business logic. This might include document_indexing.py
(DocumentProcessor & index management) and qa_pipeline.py (or similar) that orchestrates
retrieval + LLM answering.
llm/ : for LLM related modules (the unified llm_service.py and prompt templates).
retrieval/ : for retrieval logic (the unified retrieval system class and possibly BM25 utilities).
ui/ : keep UI components and styles here (already exists).
error_handling/ : keep error logging and decorators (already structured as such). This can be
implemented by creating these subfolders (if not already), moving the modules into them, and
adjusting import statements throughout the code (or using the existing utils subpackages like
utils.index_management , utils.prompts , etc., but possibly renaming utils to something
like workapp or app for clarity). Decoupling into packages also means each module has a
focused purpose, improving maintainability.
Decouple Index Management from Document Processing – The DocumentProcessor class
currently handles chunking, embedding, and index I/O all together. Why: Following single-responsibility
principle will make each piece easier to test and modify (e.g. you might swap out the indexing
backend without touching chunking logic). How: Split responsibilities either via classes or helper
modules:
Extract index persistence and maintenance functions ( save_index , load_index ,
clear_index , index dimension checks, etc.) into an IndexManager class (you already have a
index_manager_unified.py outline). This IndexManager can encapsulate FAISS index
handling, saving/loading to disk, and verifying index health (dimension mismatches, freshness). The
DocumentProcessor can then use an IndexManager instance internally for those operations.
Let DocumentProcessor focus on document ingestion: loading files, splitting into chunks, and
generating embeddings for new documents. It would call IndexManager to add those
embeddings to the index or rebuild as needed. Implementing this might mean moving some
methods out of DocumentProcessor into IndexManager (for example, the
_ensure_index_dimension or _normalize_embeddings methods and the FAISS save/load
logic). Use the functions already in utils/index_management/index_operations.py as the
basis for the IndexManager’s methods to avoid duplicating code. After this separation, each class will
be shorter and clearer. The workapp3.py (or pipeline) then coordinates: e.g.
doc_proc.load_and_chunk_document(...) to ingest files,
index_manager.index_documents(...) to build the index, and
retrieval_system.query(...) to handle queries.
Isolate the QA Pipeline Logic – Currently, the Streamlit script ( workapp3.py ) likely contains logic
intermixing UI and backend calls (DocumentProcessor, LLMService, etc.). Why: Separating pure logic
from UI allows easier testing (e.g. call the QA pipeline function directly) and potential reuse (if a CLI
•
•
•
•
•
•
•
•
•
•
3
or API is added later). How: Create a function or module for the question-answering pipeline. For
example, a function answer_query(query: str) -> dict that internally:
Uses the retrieval system ( UnifiedRetrievalSystem /DocumentProcessor) to fetch relevant
context.
Calls the LLM service to get an answer.
Formats the result (combining the extracted answer and any formatting). This function can live in a
new module (e.g. core/qa_pipeline.py ). In workapp3.py , replace the inline logic with calls to
this function, and then simply display the results via the UI components. Keep workapp3.py
mostly for UI layout and user interaction handling. This decoupling means the core logic doesn’t
depend on Streamlit and could be invoked elsewhere, and it makes the Streamlit file smaller and
more focused on presentation.
Inject Configurations Cleanly – Currently, configuration is accessed via a global config_unified
module (with objects like retrieval_config , model_config , etc.). Why: While using a config
module is convenient, it can entangle modules with global state and complicate testing (it’s hard to
override configs for a single test). How: Introduce a clearer config management approach:
Load configuration from config.json and performance_config.json once (perhaps in an
app initialization section or in a Config class).
Pass relevant config sections to components that need them. For example, initialize
LLMService(api_key, model_config, performance_config) instead of having it import the
module. Similarly, DocumentProcessor could take retrieval_config or specific settings it
needs.
Alternatively, use a singleton config object or a context that components query, but ensure it's
modifiable for tests (e.g. allow overriding an env var or passing a dict to the config loader). This
change decouples the components from a hard-coded config source and makes their dependencies
explicit. It also aligns with maintainability by clearly showing which parts of the config each module
uses. If doing this is too large a change to implement immediately, at least standardize naming in
the current config_unified.py : for instance, ensure all config dataclasses and variables use
consistent naming conventions and are grouped logically (the current approach is acceptable, just
keep it well-documented and maybe rename config_unified.py to config.py now that it's
the single config file).
3. Split and Streamline Large Modules
Split the Monolithic DocumentProcessor – At ~2.3k lines, document_processor_unified.py is
quite large and multitasks. Why: Splitting it into focused units will improve readability and make it
easier to locate specific functionality (e.g. cache management vs. embedding vs. search). How:
Consider breaking this file into two or more modules:
One module or class dedicated to Document Ingestion (file loading, text extraction, chunk splitting,
and cleaning). This part handles everything up to producing cleaned text chunks with metadata. It
would include functions like _get_file_loader , load_and_chunk_document ,
_handle_small_chunks , etc., plus the caching of chunks ( ChunkCacheEntry and related cache
methods).
•
•
•
•
•
•
•
•
•
4
Another module/class for Embedding Index Management (building FAISS index, adding
embeddings, deduplicating, saving/loading index files). This could be the earlier mentioned
IndexManager or integrated into DocumentProcessor as a smaller class. Functions like
create_embeddings , _build_faiss_index , search_index , _ensure_index_dimension ,
and index save/load routines would reside here. You can have DocumentProcessor orchestrate
these pieces (or even rename DocumentProcessor to something like DocumentIndexBuilder
if its role is primarily indexing). Splitting this way results in files of manageable size (perhaps ~1000
lines each or less) and a clearer separation between “getting text data” and “managing vector
indexes.” Each new sub-module can live in the index_management/ package or a new
document/ package. Be sure to adjust imports and ensure that the integrated flow still works (for
example, after chunking, pass the chunks to the index manager to embed and add to index).
Trim Down workapp3.py – The main app script is over 1100 lines, which mixes UI components, state
management, and some logic. Why: A leaner main script is easier to understand and maintain. How:
After moving core QA logic out (as described above), apply these refactoring steps to
workapp3.py :
Modularize UI sections: If there are distinct UI sections (e.g. sidebar config vs. main Q&A interface),
consider splitting some of that into functions or moving into the ui/ module. For instance, you
might create ui/layout.py containing functions like setup_sidebar(config) or
render_chat_interface(...) which build portions of the UI. workapp3.py would simply call
those, making the structure of the app more evident.
Remove dead code or logs: Ensure any old references to removed modules (like if workapp3.py
still imports enhanced_retrieval or similar) are cleaned up. Also consider moving verbose
logging or debugging info to a separate debug panel or remove it if not needed.
Use the unified components: Replace calls to any outdated UI helper with the new consolidated
ones (e.g. use display_answer now enhanced with new features, instead of a mix of old/new
functions). These changes will reduce the file size and complexity. After refactoring, workapp3.py
should primarily handle Streamlit interactions (file uploader, form submission, button clicks) and
delegate everything else to the appropriate module or function.
Right-Size Other Modules – Identify any other overly large files and logically split them. For instance:
The unified LLMService module (~800+ lines) could be split if needed: perhaps separate out the
caching mechanism or the asynchronous vs. synchronous logic. However, given it revolves around
one class, it's acceptable in one file. Just ensure internal helper functions are well-organized (you
might group related methods together and add comments or docstrings for sections like “Caching
Utilities” vs “LLM Call Methods” for clarity).
The error_logging.py and logging_standards.py are small, but note that
logging_standards.py seems to serve as documentation/examples. You might not need to split
anything here, but consider moving example or documentation content into the project README or
a docs folder to keep code modules focused.
The Prompts package: if it grows complex, ensure each prompt type stays in its own file (which is
already done: extraction vs. formatting vs. system message). No further splitting needed, just
maintain that pattern as you add or remove prompts. In summary, aim for modules to stay under a
•
•
•
•
•
•
•
•
•
5
few hundred lines each if possible, focusing on a specific theme. Where a file naturally grows too
large, it's a sign to break it into sub-components or helper modules.
4. Consistent Naming and Style
Drop the “unified/enhanced” prefixes – After merging duplicate modules, the naming can be
simplified. Why: Names like *_unified.py or *_enhanced.py were to distinguish transitional
versions; with only one version remaining, those qualifiers add noise. How: Rename files and classes
to intuitive names:
config_unified.py → config.py (since it's now the single configuration module).
document_processor_unified.py → perhaps split as suggested, or if kept as one, call it
document_processor.py or index_builder.py depending on its role.
unified_retrieval_system.py → maybe just retrieval_system.py or even
retrieval.py .
Remove “enhanced” from any remaining file names (e.g. formatting_prompt_enhanced.py will
be gone, llm_service_enhanced.py gone, etc.). The classes inside can also drop these
adjectives; for example, just LLMService instead of differentiating an enhanced one. Ensure you
update import statements throughout the project to match the new names. This makes the module
names reflect their purpose without referencing the old vs new distinction that no longer exists.
Apply Consistent Naming Conventions – Ensure functions, variables, and classes follow a uniform
style across the codebase. Why: Consistency improves readability and helps avoid mistakes. How:
Review the code for any inconsistencies or legacy naming:
Functions and methods should use lower_snake_case , and classes use UpperCamelCase (this
seems mostly followed already).
Check naming of similar functions: e.g., the project has both handle_small_chunks and
_handle_small_chunks (one as free function, one as method). After unification, there should be
only one; ensure its name clearly indicates its purpose and is used uniformly.
Ensure config field names are clear and consistent (if some config keys or dataclass fields had
different naming schemes, normalize them). For example, if model_config contains
max_tokens while another place uses maxTokens , convert to one style (preferably snake_case
for Python code).
Standardize logging messages and error terminology as per your logging standards. For instance,
decide on using either “Error:” prefix in returned error messages everywhere or not at all, consistent
casing for warnings, etc. (The provided logging_standards.py can guide these choices).
Remove any leftover commented-out code or TODO markers that were relevant to the old duplicate
code. Since you now have a separate Errors_Roadmap.md and presumably a Git history, you can
clean out commented blocks (such as progress comments like “# Next pending LLM-1”) to declutter
the code. Keep the codebase tidy with only active, relevant code or clearly marked future tasks.
Follow PEP8 Formatting – As a final pass, ensure the code style is uniform. Why: A consistent style
makes the codebase look professional and avoids trivial diffs in version control. How: Use linters/
formatters to standardize:
•
•
•
•
•
•
•
•
•
•
•
•
6
Run a PEP8 linter (or tools like flake8 / pylint ) to catch any remaining style issues (long lines,
extra whitespace, etc.). For instance, large dictionaries or JSON schemas could be pretty-formatted
for readability, long exception handling blocks might be split into smaller lines, etc.
Apply a formatter (like black with an appropriate line-length setting) to automatically format the
code. This will handle indentation, spacing around operators, and wrapping long lines in a consistent
way. Pay special attention to the large modules after splitting; formatting them can significantly
improve readability.
Ensure docstrings are present for all public classes and functions, and update them if any function
signatures changed during refactoring. For example, if you merged functions or changed
parameters (like no longer passing chunk_size into load_and_chunk_document because it
uses the internal config), reflect that in the docstring. Consistent triple-quote docstring style and
tone should be used throughout.
Check that naming changes are reflected everywhere (including in comments and documentation
files). For example, if unified_retrieval_system.py became retrieval_system.py , update
README or any usage examples accordingly.
By performing the above steps in order (deduplicating first, then restructuring, then splitting and renaming,
and finally styling), you'll incrementally improve the codebase while keeping it functional at each stage.
Each group of changes addresses a specific maintainability goal: eliminate redundant code, organize
logically, keep files concise, and ensure consistency. Following this refactoring plan will result in a
cleaner, more modular WorkApp2 codebase that is easier to understand and extend, even for an LLM in VS
Code with CLine assistance.
•
•
•
•
7