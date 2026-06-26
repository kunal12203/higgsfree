pipeline {
    agent any

    // Only build PRs that have the "approved-for-ci" label (set by maintainer)
    options {
        disableConcurrentBuilds()
    }

    environment {
        GITHUB_TOKEN = credentials('github-token')
        TEST_VIDEO   = '/home/ubuntu/ci_test/sample.MOV'
        WORK_DIR     = "/home/ubuntu/ci_test/run_${env.BUILD_NUMBER}"
        OUTPUT_VIDEO = "/home/ubuntu/ci_test/run_${env.BUILD_NUMBER}/output.mp4"
        TEST_TEXT    = 'Hey! This is a quick test of the video generation pipeline.'
    }

    stages {

        stage('Check approval') {
            steps {
                script {
                    if (env.CHANGE_ID) {
                        def labels = sh(
                            script: """
                                curl -s -H "Authorization: token ${env.GITHUB_TOKEN}" \
                                  "https://api.github.com/repos/kunal12203/higgsfree/issues/${env.CHANGE_ID}/labels" \
                                  | grep -o '"approved-for-ci"' | head -1
                            """,
                            returnStdout: true
                        ).trim()
                        if (!labels) {
                            error("PR does not have 'approved-for-ci' label — skipping CI. A maintainer must add the label to approve.")
                        }
                        echo "PR approved for CI — proceeding"
                    }
                }
            }
        }

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Get baseline score') {
            steps {
                script {
                    env.BASELINE = sh(
                        script: "cat /home/ubuntu/ci_test/baseline_score.txt 2>/dev/null || echo '0.0'",
                        returnStdout: true
                    ).trim()
                    echo "Baseline score: ${env.BASELINE}"
                }
            }
        }

        stage('Run pipeline') {
            steps {
                sh """
                    mkdir -p ${env.WORK_DIR}
                    /home/ubuntu/venv-sonic/bin/python worker/test_seated_pipeline.py \
                        ${env.TEST_VIDEO} \
                        ${env.OUTPUT_VIDEO} \
                        ${env.WORK_DIR} \
                        --text '${env.TEST_TEXT}' \
                        --scene studio --aspect 9:16
                """
            }
        }

        stage('Score output') {
            steps {
                script {
                    env.SCORE = sh(
                        script: """
                            /home/ubuntu/venv-sonic/bin/python eval/score_pipeline.py \
                                ${env.TEST_VIDEO} \
                                ${env.OUTPUT_VIDEO}
                        """,
                        returnStdout: true
                    ).trim()
                    echo "Pipeline score: ${env.SCORE}"
                }
            }
        }

        stage('Post PR comment') {
            when {
                expression { env.CHANGE_ID != null }
            }
            steps {
                script {
                    def baseline = env.BASELINE.toFloat()
                    def score    = env.SCORE.toFloat()
                    def diff     = score - baseline
                    def arrow    = diff >= 0 ? "▲" : "▼"
                    def status   = diff >= 0 ? "✅" : "❌"
                    def diffStr  = String.format("%.3f", Math.abs(diff))
                    def scoreStr = String.format("%.3f", score)
                    def baseStr  = String.format("%.3f", baseline)

                    def body = """## ${status} Pipeline Quality Score

| Metric | Score |
|--------|-------|
| This PR | **${scoreStr}** |
| Baseline (main) | ${baseStr} |
| Diff | ${arrow} ${diffStr} |

**Score breakdown:** face identity similarity (50%) + lip sync confidence (50%)

_Tested on g5.2xlarge (A10G 24GB) · Build #${env.BUILD_NUMBER}_"""

                    sh """
                        curl -s -X POST \
                          -H "Authorization: token ${env.GITHUB_TOKEN}" \
                          -H "Content-Type: application/json" \
                          -d '{"body": ${groovy.json.JsonOutput.toJson(body)}}' \
                          "https://api.github.com/repos/kunal12203/higgsfree/issues/${env.CHANGE_ID}/comments"
                    """
                }
            }
        }

        stage('Update baseline') {
            when {
                branch 'main'
            }
            steps {
                sh "echo ${env.SCORE} > /home/ubuntu/ci_test/baseline_score.txt"
                echo "Baseline updated to ${env.SCORE}"
            }
        }
    }

    post {
        always {
            sh "rm -rf ${env.WORK_DIR} || true"
        }
    }
}
