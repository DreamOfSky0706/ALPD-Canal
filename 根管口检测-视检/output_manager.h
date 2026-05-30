#pragma once
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// 文件输出我规定完格式就交给AI了，反正结果来看做得还不赖。
class OutputManager
{
public:
	explicit OutputManager(const std::string& rootDir = "output");

	std::string createImageDir(const std::string& imagePath);

	std::string getPath(const std::string& dir, const std::string& filename);

	std::string rootDir() const
	{
		return rootDir_;
	}

private:
	std::string rootDir_;
};