param(
    [Parameter(Mandatory=$true)]$Owner,
    [Parameter(Mandatory=$true)]$Repo,
    [Parameter(Mandatory=$true)]$PrNumber,
    [string]$Token
)

# AtomGit PR 信息获取脚本
# 用法: .\fetch_pr.ps1 <owner> <repo> <pr_number>
# 示例: .\fetch_pr.ps1 -Owner "mindspore" -Repo "mindspore" -PrNumber 92548
#
# 首次运行会提示输入token，之后会保存到本地

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TokenFile = Join-Path $ScriptDir ".atomgit_token"

function Get-Token {
    if (-not [string]::IsNullOrWhiteSpace($Token)) {
        # 如果命令行提供了Token，保存它
        $tokenObj = @{
            value = $Token
            expiry = (Get-Date).AddDays(30).ToString("yyyy-MM-dd")
        }
        $tokenObj | ConvertTo-Json | Set-Content $TokenFile
        Write-Host "Token已保存到本地 (有效期30天)"
        return $Token
    }
    if (Test-Path $TokenFile) {
        $token = Get-Content $TokenFile -Raw | ConvertFrom-Json
        $expiry = [DateTime]::ParseExact($token.expiry, "yyyy-MM-dd", $null)
        if ($expiry -gt (Get-Date)) {
            Write-Host "使用已保存的Token (有效期至 $($token.expiry))"
            return $token.value
        }
    }
    Write-Host "请输入 AtomGit 私人令牌 (PRIVATE-TOKEN): " -NoNewline
    $inputToken = Read-Host
    if ([string]::IsNullOrWhiteSpace($inputToken)) {
        Write-Host "Token不能为空"
        exit 1
    }
    $tokenObj = @{
        value = $inputToken
        expiry = (Get-Date).AddDays(30).ToString("yyyy-MM-dd")
    }
    $tokenObj | ConvertTo-Json | Set-Content $TokenFile
    Write-Host "Token已保存到本地 (有效期30天)"
    return $inputToken
}

$Token = Get-Token
$BaseUrl = "https://api.atomgit.com/api/v5/repos/${Owner}/${Repo}/pulls/${PrNumber}"

Write-Host "=== PR 详情 ==="
try {
    $pr = Invoke-RestMethod -Uri $BaseUrl -Headers @{'PRIVATE-TOKEN'=$Token;'Accept'='application/json'}
    $pr | ConvertTo-Json -Depth 10
} catch {
    Write-Host "获取PR详情失败: $_"
    Remove-Item $TokenFile -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== PR 文件列表 ==="
try {
    $files = Invoke-RestMethod -Uri "$BaseUrl/files" -Headers @{'PRIVATE-TOKEN'=$Token;'Accept'='application/json'}
    $files | ConvertTo-Json -Depth 10
} catch {
    Write-Host "获取PR文件列表失败: $_"
}