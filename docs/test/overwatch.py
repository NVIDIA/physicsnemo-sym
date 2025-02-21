from git import Repo
from pathlib import Path
import datetime
from pytz import timezone
import re
import gitlab
import argparse


def get_commit_files(
    repo_path: Path, branch_name: str, day_range: int = 1, max_count: int = 100
):
    """Gets a list for files changed in commits for past number of days

    Parameters
    ----------
    repo_path : Path
        Path to git repo
    branch_name : str
        Branch to use for commits
    day_range : int, optional
        Number of past days to look for commits, by default 1
    max_count : int, optional
       Max number of commits to look at , by default 100

    Returns
    -------
    Dict[str, Tuple[int, str]]
        Dictionary of changes files with the latest commit time and hash
    """
    assert repo_path.is_dir(), "Invalid repo folder path"
    repo = Repo(repo_path)
    assert not repo.bare, "Repo is bare"
    assert branch_name in repo.heads, "Branch name {} not found in available heads"
    branch = repo.heads[branch_name]
    files = {}
    # Iterates from newest to oldest commit
    for commit in list(
        repo.iter_commits(
            rev=branch, since=f"{day_range}.days.ago", max_count=max_count
        )
    ):
        for file in commit.stats.files.keys():
            if file not in files:
                files[file] = tuple([commit.committed_date, commit.hexsha])

    return files


def get_doc_codeblock_files(userguide_path: Path, file_pattern: str = "*.rst"):
    """Looks through RST files for any references to example python files

    Parameters
    ----------
    userguide_path : Path
        Path to user guide RST files
    file_pattern : str, optional
        Pattern for file types to parse, by default "*.rst"

    Returns
    -------
    Dict[str, Dict[str, List[int]]]
        Dictionary of python files that are contained in doc files with line numbers.
        Returned dictionary maps between python file to a dictionary containing each
        documentation file and line numbers it is referenced.
    """
    assert userguide_path.is_dir(), "Invalid repo folder path"

    regex_pattern = re.compile("\/physicsnemo\/examples\/(.+?)\.py")

    files = {}
    for doc_file in userguide_path.rglob(file_pattern):
        for i, line in enumerate(open(doc_file)):
            for match in re.finditer(regex_pattern, line):
                python_file = str(Path(*Path(match.group()).parts[3:]))
                doc_file_local = str(Path(*Path(doc_file).parts[1:]))
                if not python_file in files:
                    files[python_file] = {str(doc_file_local): [i + 1]}
                else:
                    if doc_file_local in files[python_file]:
                        files[python_file][doc_file_local].append(i + 1)
                    else:
                        files[python_file][doc_file_local] = [i + 1]

    return files


def create_gitlab_issue(commit_files, doc_files, gl_token: str):
    """Creates a Gitlab issue if changed files are present

    Parameters
    ----------
    commit_files : Dict[str, Tuple[int, str]]
        Dictionary of changes files with the latest commit time and hash
    doc_files : Dict[str, Dict[str, List[int]]]
        Dictionary of python files that are contained in doc files with line numbers.
    gl_token : str
        Gitlab API access token, should be passed in via program arguement
        (do not hard code!)
    """
    # .git urls
    examples_repo_url = (
        "https://gitlab-master.nvidia.com/simnet/examples/-/blob/develop/"
    )
    docs_repo_url = "https://gitlab-master.nvidia.com/simnet/docs/-/blob/develop/"
    ug_folder = "user_guide/"

    def file_desc(file_name, commit_time_stamp, commit_hash, doc_files):
        # Create description string for one updated file
        # Convert time-stamp to string in pacific time
        commit_time = datetime.datetime.fromtimestamp(commit_time_stamp)
        commit_time = commit_time.astimezone(timezone("US/Pacific"))

        desc_str = f"---\n\n"
        desc_str += f"[{file_name}]({examples_repo_url}{file_name})\n\n"
        desc_str += f":date: Editted: {commit_time.strftime('%Y-%m-%d %H:%M PST')}\n\n"
        desc_str += f":fox: Commit: simnet/examples@{commit_hash[:8]}\n\n"
        desc_str += ":mag: Files to check:\n"

        for doc_file in doc_files.keys():
            doc_file = Path(doc_file)
            desc_str += f"- {doc_file.name} : "
            for line in doc_files[str(doc_file)]:
                desc_str += (
                    f"[L{line}]({docs_repo_url}{ug_folder}{str(doc_file)}#L{line}), "
                )
            desc_str = desc_str[:-2]
            desc_str += "\n\n"

        return desc_str

    todays_date = datetime.date.today()
    issue_title = f"[Overwatch] Example files updated {todays_date.month}/{todays_date.day}/{todays_date.year}"
    issue_desc = "### :robot: Overwatch Detected Files:\n\n"
    issue_desc += "This is an automated issue created by CI Example Overwatch bot.\n\n"
    changed_files = False

    # Loop over changed files in detected commits
    for commit_file_name in commit_files.keys():
        if commit_file_name in doc_files:
            issue_desc += file_desc(
                commit_file_name,
                commit_files[commit_file_name][0],
                commit_files[commit_file_name][1],
                doc_files[commit_file_name],
            )
            changed_files = True

    # If no updated files just return
    if not changed_files:
        print("No updated files detected.")
        return
    else:
        print("File changes detected, creating issue.")

    # Log into gitlab and create issue
    gl = gitlab.Gitlab("https://gitlab-master.nvidia.com", private_token=gl_token)
    p = gl.projects.get("simnet/docs")
    p.issues.create(
        {
            "title": issue_title,
            "description": issue_desc,
            "labels": ["user guide", "update"],
        }
    )


if __name__ == "__main__":
    # This should be ran outside of the doc repo directory (1 level up)
    p = argparse.ArgumentParser()
    p.add_argument("--gitlab-token", type=str, default=None, help="Gitlab API token")
    p.add_argument(
        "--day-range", type=int, default=1, help="Day range to check commits"
    )
    args = vars(p.parse_args())
    # Paths inside CI docker container
    user_guide_path = Path("./user_guide")
    example_repo_path = Path("./external/examples")

    print("Parsing .rst files for python references")
    doc_files = get_doc_codeblock_files(user_guide_path)

    print("Checking examples repo for recent commits")
    commit_files = get_commit_files(
        example_repo_path, "develop", day_range=args["day_range"]
    )

    print("Checking for relevant file changes")
    create_gitlab_issue(commit_files, doc_files, args["gitlab_token"])
