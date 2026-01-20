def create_submission(executable:str, njobs:int, arguments:list, log_dir:str, input_files:list, file_name:str="run.sub", disk_space:int=200, memory:int=2, cpus:int=2):
    arguments_contents = ' '.join(map(str, arguments))
    input_files_contents = ' '.join(map(str, input_files))
    with open(file_name, 'w') as file:
        submit_lines = f"""Universe = vanilla
+IsLocalJob = true
request_disk = {disk_space}MB
request_memory = {memory}GB
request_cpus = {cpus}
hold = False
executable              = {executable}
arguments               = {arguments_contents}
log                     = {log_dir}/log_$(PROCESS).log
output                  = {log_dir}/out_$(PROCESS).txt
error                   = {log_dir}/error_$(PROCESS).txt
should_transfer_files   = Yes
when_to_transfer_output = ON_EXIT
+isSmallJob = true
transfer_input_files = {input_files_contents}
getenv = true
queue {njobs}
        """
        file.write(submit_lines)

if __name__ == "__main__":
    create_submission("test.py", njobs=100, arguments=[1,2,3], log_dir="/home/ryan/", input_files=['test2.py', 'test3.py'])
