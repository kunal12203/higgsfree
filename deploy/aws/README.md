# AWS asset-bucket privacy notes (`avatar-graperoot-assets`)

The CDN (`d1ikua6hus1yzq.cloudfront.net`, distribution `E3EF94Q0FLT4JS`) fronts
the **entire** `avatar-graperoot-assets` bucket. By default that made raw PII
(consent videos, voice-clone audio) publicly downloadable by anyone who knew the
object path. We lock those prefixes down with explicit `Deny` statements scoped
to the CloudFront service principal (the app's own IAM access is unaffected).

## Prefix classification

| Prefix | Contains | CDN access |
|--------|----------|------------|
| `videos/`, `broll-cache/`, `deploy/`, `test/` | rendered outputs / b-roll / static | public (OK) |
| `avatars/*/left-*.jpg`, `avatars/*/right-*.jpg`, `avatars/*/portrait.png` | display imagery | public (OK) |
| `consent-videos/*` | consent / source clips | **DENIED** (PII) |
| `avatars/*/consent-*` | per-avatar consent video | **DENIED** (PII) |
| `avatars/*/voice_profile/*` | cloned-voice reference audio | **DENIED** (biometric) |
| `avatars/*/face.jpg` | extracted face capture | **DENIED** (PII) — show via presigned token, see below |

## Rollback

The ORIGINAL (pre-change) bucket policy is saved next to this file as
`avatar-graperoot-assets.policy.original.json`. To fully restore the original
(removes ALL the privacy Deny statements — only do this if delivery breaks):

```bash
aws s3api put-bucket-policy --bucket avatar-graperoot-assets \
  --policy file://deploy/aws/avatar-graperoot-assets.policy.original.json
```

## Current applied policy

See `avatar-graperoot-assets.policy.current.json` for the policy currently in
effect (Allow + Deny statements).

## face.jpg — show via presigned token, not the public CDN

`face.jpg` is **not** rendered anywhere in the UI today (the avatar preview uses
`cachedFrameKey` = `portrait.png`). It's an internal artifact + a DB key
(`frontImageKey`) used server-side. So it is now DENIED on the public CDN.

If you ever need to display the face to an authorized user, do NOT use a CDN
URL — mint a short-lived presigned S3 URL (the "token"). The app already has the
helper (`src/lib/s3.ts → getPresignedDownloadUrl`). Example endpoint:

```ts
// src/app/api/avatars/[id]/face/route.ts
import { getPresignedDownloadUrl } from '@/lib/s3';
// ...auth-check the session/owner first...
const url = await getPresignedDownloadUrl(`avatars/${id}/face.jpg`); // 1h token
return Response.json({ url });
```

The presigned URL hits S3 directly (bypasses CloudFront), is time-limited, and
is only handed to an authenticated, authorized caller.

## Optional: also lock the raw capture photos

`avatars/*/left-*.jpg`, `right-*.jpg`, `front-*.jpg`, `face-cropped.jpg` are also
raw face captures (PII) and are not displayed in the UI either (only
`portrait.png` is). They are currently still public on the CDN. Consider adding
them to the Deny list if you want zero raw-face exposure.
