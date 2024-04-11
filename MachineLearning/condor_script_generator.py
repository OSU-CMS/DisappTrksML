class HTCondorScriptGenerator:
    def __init__(self, executable, output_file="out_$(PROCESS).txt", error_file="error_$(PROCESS).txt", log_file="log_$(PROCESS).log", log_dir="."):
        self.script_lines = []
        self.script_lines.append("Executable = {}".format(executable))
        self.script_lines.append("Output = {}/{}".format(log_dir, output_file))
        self.script_lines.append("Error = {}/{}".format(log_dir, error_file))
        self.script_lines.append("Log = {}/{}".format(log_dir,log_file))

    def add_argument(self, argument):
        self.script_lines.append("Arguments = {}".format(argument))

    def add_requirements(self, requirements):
        self.script_lines.append("Requirements = {}".format(requirements))

    def add_queue(self, num_jobs=1):
        self.script_lines.append("Queue {}".format(num_jobs))

    def add_option(self, option:str, value:str):
        self.script_lines.append("{} = {}".format(option, value))

    def generate_script(self, file_path):
        with open(file_path, "w") as file:
            file.write("\n".join(self.script_lines))

# Example usage:
if __name__ == "__main__":
    # Create an instance of HTCondorScriptGenerator
    condor_generator = HTCondorScriptGenerator(executable="my_executable")

    # Add arguments, requirements, and specify the number of jobs to queue
    condor_generator.add_argument("arg1 arg2 arg3")
    condor_generator.add_requirements('OpSysMajorVer == 7')
    condor_generator.add_option("should_transfer_files", "yes")

    condor_generator.add_queue(num_jobs=5)
    # Generate the HTCondor script and save it to a file
    condor_generator.generate_script("my_condor_script.submit")
